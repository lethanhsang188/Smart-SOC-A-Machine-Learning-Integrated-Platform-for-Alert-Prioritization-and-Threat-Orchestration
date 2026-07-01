"""Convert Wazuh Discover CSV export to JSON-lines.

The Wazuh dashboard exports flattened columns such as:
  _source.rule.id, _source.rule.level, _source.agent.name, _source.data.url

This script can emit either:
  - raw: JSON-lines shaped like /var/ossec/logs/alerts/alerts.json
  - training: the older normalized JSON shape used by this project
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import unquote_plus


APACHE_RE = re.compile(
    r'"(?P<method>GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\s+(?P<url>\S+)\s+HTTP/[\d.]+"\s+(?P<status>\d{3})\s+(?P<size>\d+|-)(?:\s+"[^"]*"\s+"(?P<ua>[^"]*)")?',
    re.IGNORECASE,
)


def clean(value: Any) -> str:
    if value is None:
        return ""
    value = str(value)
    return "" if not value.strip() else value.strip()


def clean_value(value: Any) -> Any:
    text = clean(value)
    if text == "":
        return ""
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if text.startswith(("[", "{")):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    return text


def to_int(value: Any, default: int = 0) -> int:
    try:
        value = clean(value)
        return int(float(value)) if value else default
    except ValueError:
        return default


def parse_jsonish_list(value: Any) -> List[str]:
    text = clean(value)
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(x) for x in data]
    except json.JSONDecodeError:
        pass
    return [x.strip() for x in re.split(r"[,;]", text.strip("[]")) if x.strip()]


def parse_timestamp(value: Any) -> str:
    """Convert Wazuh Dashboard timestamps to raw alerts.json timestamp format."""
    text = clean(value)
    if not text:
        return ""
    for fmt in ("%b %d, %Y @ %H:%M:%S.%f", "%b %d, %Y @ %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "+0000"
        except ValueError:
            pass
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "+0000"
    except ValueError:
        return text


def set_nested(target: Dict[str, Any], dotted_path: str, value: Any) -> None:
    cur = target
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


def nested_from_prefix(row: Dict[str, str], prefix: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, raw_value in row.items():
        if not key.startswith(prefix):
            continue
        value = clean_value(raw_value)
        if value == "":
            continue
        set_nested(result, key[len(prefix):], value)
    return result


def add_list_field(target: Dict[str, Any], key: str, value: Any) -> None:
    items = parse_jsonish_list(value)
    if items:
        target[key] = items


def prune_empty(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: prune_empty(v) for k, v in value.items() if prune_empty(v) not in ("", [], {}, None)}
    if isinstance(value, list):
        return [prune_empty(v) for v in value if prune_empty(v) not in ("", [], {}, None)]
    return value


def prune_raw_alert(alert: Dict[str, Any]) -> Dict[str, Any]:
    """Prune optional empty values while preserving raw Wazuh top-level keys."""
    required_top = {"timestamp", "rule", "agent", "manager", "id", "full_log", "decoder", "location"}
    pruned = prune_empty(alert)
    for key in required_top:
        if key not in pruned and key in alert:
            pruned[key] = alert[key]
    return pruned


def parse_apache_full_log(full_log: str) -> Dict[str, Any]:
    full_log = clean(full_log)
    match = APACHE_RE.search(full_log)
    if not match:
        return {}
    size_text = match.group("size")
    return {
        "method": match.group("method").upper(),
        "url": match.group("url"),
        "status": to_int(match.group("status")),
        "user_agent": match.group("ua") or "",
        "bytes_toclient": 0 if size_text == "-" else to_int(size_text),
    }


def infer_attack_type(rule_id: str, description: str, url: str, groups: Iterable[str], full_log: str) -> str:
    text = " ".join([rule_id, description, url, full_log, " ".join(groups)]).lower()
    decoded = unquote_plus(text)

    if rule_id in {"100415", "2410030"} or "/vulnerabilities/exec/" in decoded or any(x in decoded for x in ["command_injection", "cmdi", "whoami", "/etc/passwd", " uname "]):
        return "command_injection"
    if rule_id in {"100413", "100420", "100421", "2410010", "31164"} or "/vulnerabilities/sqli/" in decoded or "sql injection" in decoded or "sqlinjection" in decoded:
        return "sql_injection"
    if rule_id in {"100414", "100422", "2410020"} or "/vulnerabilities/xss_d/" in decoded or "/vulnerabilities/xss_r/" in decoded or any(x in decoded for x in ["<script", "xss", "onerror", "document.cookie"]):
        return "xss"
    if rule_id in {"100412", "100424", "2410040"} or "/vulnerabilities/brute/" in decoded or "bruteforce" in decoded:
        return "bruteforce"
    if rule_id in {"100410", "2410070"} or any(x in decoded for x in ["syn flood", "ddos", "dos"]):
        return "dos"
    if rule_id in {"100411", "2410080"} or ("ssh" in decoded and "brute" in decoded):
        return "ssh_bruteforce"
    if "mimikatz" in decoded or "mimilib" in decoded:
        return "malware_tool"
    return "normal"


def derive_priority(rule_id: str, rule_level: int, attack_type: str, description: str, groups: Iterable[str]) -> str:
    group_text = " ".join(groups).lower()
    desc = description.lower()

    if attack_type in {"command_injection", "dos"}:
        return "P1"
    if attack_type == "sql_injection":
        return "P1"
    if attack_type in {"xss", "ssh_bruteforce"}:
        return "P2"
    if attack_type in {"bruteforce", "malware_tool"}:
        return "P3"

    if rule_level >= 12:
        return "P1"
    if rule_level >= 10:
        return "P2"
    if rule_level >= 7:
        return "P3"

    if any(x in group_text for x in ["attack", "authentication_failed", "ids"]):
        return "P3"
    if any(x in desc for x in ["failed", "invalid", "denied"]):
        return "P3"
    return "P4"


def normalize_row(row: Dict[str, str]) -> Dict[str, Any]:
    rule_id = clean(row.get("_source.rule.id"))
    rule_level = to_int(row.get("_source.rule.level"))
    rule_description = clean(row.get("_source.rule.description"))
    groups = parse_jsonish_list(row.get("_source.rule.groups"))
    full_log = clean(row.get("_source.full_log"))
    apache = parse_apache_full_log(full_log)

    data_url = (
        clean(row.get("_source.data.http.url"))
        or clean(row.get("_source.data.url"))
        or apache.get("url", "")
    )
    decoded_url = unquote_plus(data_url)
    method = (
        clean(row.get("_source.data.http.http_method"))
        or clean(row.get("_source.data.protocol"))
        or clean(row.get("_source.data.proto"))
        or apache.get("method", "")
    )
    http_status = to_int(row.get("_source.data.http.status")) or apache.get("status", 0)
    user_agent = clean(row.get("_source.data.http.http_user_agent")) or apache.get("user_agent", "")

    srcip = (
        clean(row.get("_source.data.src_ip"))
        or clean(row.get("_source.data.srcip"))
        or clean(row.get("_source.data.flow.src_ip"))
        or clean(row.get("_source.srcip"))
        or clean(row.get("_source.data.win.eventdata.ipAddress"))
    )
    dstip = (
        clean(row.get("_source.data.dest_ip"))
        or clean(row.get("_source.data.dstip"))
        or clean(row.get("_source.data.flow.dest_ip"))
    )

    alert_sig_id = clean(row.get("_source.alert.signature_id")) or clean(row.get("_source.data.alert.signature_id"))
    alert_sig = clean(row.get("_source.alert.signature")) or clean(row.get("_source.data.alert.signature"))
    alert_sev = to_int(row.get("_source.alert.severity") or row.get("_source.data.alert.severity"))
    alert_action = clean(row.get("_source.alert.action") or row.get("_source.data.alert.action"))

    attack_type = infer_attack_type(rule_id, rule_description, decoded_url, groups, full_log)
    priority = derive_priority(rule_id, rule_level, attack_type, rule_description, groups)
    is_attack = attack_type != "normal" or "attack" in [g.lower() for g in groups]

    return {
        "@timestamp": clean(row.get("_source.@timestamp")) or clean(row.get("_source.timestamp")),
        "id": clean(row.get("_source.id")) or clean(row.get("_id")),
        "agent": {
            "id": clean(row.get("_source.agent.id")),
            "name": clean(row.get("_source.agent.name")),
            "ip": clean(row.get("_source.agent.ip")),
        },
        "manager": {"name": clean(row.get("_source.manager.name"))},
        "rule": {
            "id": rule_id,
            "level": rule_level,
            "description": rule_description,
            "groups": groups,
            "firedtimes": to_int(row.get("_source.rule.firedtimes"), 1),
            "mitre": {
                "id": parse_jsonish_list(row.get("_source.rule.mitre.id")),
                "technique": parse_jsonish_list(row.get("_source.rule.mitre.technique")),
                "tactic": parse_jsonish_list(row.get("_source.rule.mitre.tactic")),
            },
        },
        "decoder": {"name": clean(row.get("_source.decoder.name"))},
        "srcip": srcip,
        "dest_ip": dstip,
        "message": full_log or rule_description,
        "full_log": full_log,
        "location": clean(row.get("_source.location")),
        "http": {
            "url": decoded_url,
            "raw_url": data_url,
            "method": method or "unknown",
            "status": http_status,
            "user_agent": user_agent,
            "hostname": clean(row.get("_source.data.http.hostname")),
            "protocol": clean(row.get("_source.data.http.protocol")) or clean(row.get("_source.data.app_proto")),
        },
        "suricata_alert": {
            "signature_id": alert_sig_id,
            "signature": alert_sig,
            "severity": alert_sev,
            "action": alert_action,
            "category": clean(row.get("_source.data.alert.category")),
        } if alert_sig_id or alert_sig else None,
        "flow": {
            "bytes_toclient": to_int(row.get("_source.data.flow.bytes_toclient")) or apache.get("bytes_toclient", 0),
            "bytes_toserver": to_int(row.get("_source.data.flow.bytes_toserver")) or len(decoded_url),
            "src_port": to_int(row.get("_source.data.src_port") or row.get("_source.data.srcport") or row.get("_source.data.flow.src_port")),
            "dest_port": to_int(row.get("_source.data.dest_port") or row.get("_source.data.flow.dest_port")),
        },
        "attack_type": attack_type,
        "is_attack": is_attack,
        "priority": priority,
        "tags": list(dict.fromkeys([*groups, attack_type, priority])),
    }


def raw_row(row: Dict[str, str]) -> Dict[str, Any]:
    """Convert a flattened Wazuh CSV row to raw alerts.json-like shape."""
    rule: Dict[str, Any] = {
        "level": to_int(row.get("_source.rule.level")),
        "description": clean(row.get("_source.rule.description")),
        "id": clean(row.get("_source.rule.id")),
        "firedtimes": to_int(row.get("_source.rule.firedtimes"), 1),
    }
    mail = clean(row.get("_source.rule.mail"))
    if mail:
        rule["mail"] = mail.lower() == "true"
    for field in ("groups", "pci_dss", "gpg13", "gdpr", "gdpr_IV", "hipaa", "nist_800_53", "tsc", "cis", "cis_csc"):
        csv_key = f"_source.rule.{field}"
        out_key = field if field != "gdpr_IV" else "gdpr"
        existing = rule.get(out_key, [])
        values = parse_jsonish_list(row.get(csv_key))
        if values:
            rule[out_key] = list(dict.fromkeys([*existing, *values]))

    mitre: Dict[str, Any] = {}
    add_list_field(mitre, "id", row.get("_source.rule.mitre.id"))
    add_list_field(mitre, "technique", row.get("_source.rule.mitre.technique"))
    add_list_field(mitre, "tactic", row.get("_source.rule.mitre.tactic"))
    if mitre:
        rule["mitre"] = mitre

    agent = {
        "id": clean(row.get("_source.agent.id")),
        "name": clean(row.get("_source.agent.name")),
        "ip": clean(row.get("_source.agent.ip")),
    }
    manager = {"name": clean(row.get("_source.manager.name"))}
    decoder = nested_from_prefix(row, "_source.decoder.")
    if not decoder:
        decoder = {"name": clean(row.get("_source.decoder.name"))}

    alert: Dict[str, Any] = {
        "timestamp": parse_timestamp(row.get("_source.timestamp") or row.get("_source.@timestamp")),
        "rule": rule,
        "agent": agent,
        "manager": manager,
        "id": clean(row.get("_source.id")) or clean(row.get("_id")),
        "full_log": clean(row.get("_source.full_log")),
        "decoder": decoder,
        "location": clean(row.get("_source.location")),
    }

    predecoder = nested_from_prefix(row, "_source.predecoder.")
    if predecoder:
        alert["predecoder"] = predecoder

    data = nested_from_prefix(row, "_source.data.")
    if data:
        alert["data"] = data

    syscheck = nested_from_prefix(row, "_source.syscheck.")
    if syscheck:
        alert["syscheck"] = syscheck

    previous_output = clean(row.get("_source.previous_output"))
    if previous_output:
        alert["previous_output"] = previous_output
    previous_log = clean(row.get("_source.previous_log"))
    if previous_log:
        alert["previous_log"] = previous_log

    return prune_raw_alert(alert)


def should_keep(alert: Dict[str, Any], mode: str) -> bool:
    if mode == "all":
        return True
    if mode == "web":
        return alert.get("decoder", {}).get("name") == "web-accesslog" or bool(alert.get("http", {}).get("url"))
    if mode == "security":
        return alert.get("is_attack") or alert.get("rule", {}).get("level", 0) >= 5
    raise ValueError(f"Unknown mode: {mode}")


def should_keep_raw(alert: Dict[str, Any], mode: str) -> bool:
    if mode == "all":
        return True
    decoder_name = alert.get("decoder", {}).get("name", "")
    data = alert.get("data", {})
    groups = [str(g).lower() for g in alert.get("rule", {}).get("groups", [])]
    level = alert.get("rule", {}).get("level", 0)
    has_url = bool(data.get("url") or data.get("http", {}).get("url"))
    if mode == "web":
        return decoder_name == "web-accesslog" or has_url
    if mode == "security":
        return level >= 5 or any(g in {"attack", "ids", "sqlinjection", "web_attack", "xss"} for g in groups)
    raise ValueError(f"Unknown mode: {mode}")


def convert(input_path: Path, output_path: Path, mode: str, schema: str) -> Dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats: Counter[str] = Counter()
    priorities: Counter[str] = Counter()
    attacks: Counter[str] = Counter()
    rules: Counter[str] = Counter()

    with input_path.open("r", encoding="utf-8-sig", newline="") as fin, output_path.open("w", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        for row in reader:
            stats["input_rows"] += 1
            alert = raw_row(row) if schema == "raw" else normalize_row(row)
            keep = should_keep_raw(alert, mode) if schema == "raw" else should_keep(alert, mode)
            if not keep:
                stats["skipped_rows"] += 1
                continue
            fout.write(json.dumps(alert, ensure_ascii=False, separators=(",", ":")) + "\n")
            stats["output_rows"] += 1
            if schema == "training":
                priorities[alert["priority"]] += 1
                attacks[alert["attack_type"]] += 1
            rules[str(alert["rule"]["id"])] += 1

    return {
        "input": str(input_path),
        "output": str(output_path),
        "mode": mode,
        "schema": schema,
        "stats": dict(stats),
        "priority_counts": dict(priorities),
        "attack_type_counts": dict(attacks),
        "top_rule_ids": dict(rules.most_common(20)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Wazuh Discover CSV export to JSON-lines")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/wazuh_alerts_raw.json")
    parser.add_argument("--mode", choices=["all", "security", "web"], default="all", help="Rows to keep for training")
    parser.add_argument("--schema", choices=["raw", "training"], default="raw", help="Output schema")
    parser.add_argument("--stats-out", default="")
    args = parser.parse_args()

    summary = convert(Path(args.input), Path(args.output), args.mode, args.schema)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.stats_out:
        Path(args.stats_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.stats_out).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
