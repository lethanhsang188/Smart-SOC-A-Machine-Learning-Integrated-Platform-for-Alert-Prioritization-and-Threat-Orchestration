"""Feature extraction for Wazuh alert prioritization.

This module turns Wazuh alerts into stable tabular features. It accepts both
the older normalized training schema and raw /var/ossec/logs/alerts/alerts.json
events so training and inference can use the same extractor.
"""
from __future__ import annotations

import ipaddress
import re
from typing import Any, Dict, Iterable, List
from urllib.parse import unquote_plus


ATTACK_KEYWORDS = {
    "sql_injection": ["sql", "sqli", "union", "select", "or 1=1", "information_schema"],
    "xss": ["xss", "<script", "onerror", "javascript:"],
    "command_injection": [
        "cmd=",
        "cmdi",
        "command injection",
        "command execution",
        "linux command execution",
        "exec",
        "whoami",
        "/bin/sh",
        "powershell",
        "bash",
        "nc -e",
    ],
    "path_traversal": ["../", "..\\", "etc/passwd", "win.ini"],
    "bruteforce": ["brute", "authentication_failed", "failed password", "invalid user"],
    "ssh_bruteforce": ["ssh_bruteforce", "ssh brute", "failed password", "invalid user"],
    "dos": ["dos", "ddos", "syn flood", "flood"],
    "webshell": ["webshell", ".php", "shell"],
    "malware_tool": ["mimikatz", "mimilib", "meterpreter"],
}

ATTACK_PRIORITY = {
    "command_injection": 10,
    "webshell": 10,
    "sql_injection": 9,
    "path_traversal": 8,
    "dos": 8,
    "xss": 7,
    "ssh_bruteforce": 7,
    "bruteforce": 6,
    "malware_tool": 6,
}

APACHE_RE = re.compile(
    r'"(?P<method>GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\s+(?P<url>\S+)\s+HTTP/[\d.]+"\s+(?P<status>\d{3})\s+(?P<size>\d+|-)(?:\s+"[^"]*"\s+"(?P<ua>[^"]*)")?',
    re.IGNORECASE,
)

SNORT_RE = re.compile(
    r"\[\*\*\]\s+\[(?P<signature_id>\d+:\d+:\d+)\]\s+(?P<signature>.*?)\s+\[\*\*\]",
    re.IGNORECASE,
)

RAW_RULE_ATTACK_TYPES = {
    "100415": "command_injection",
    "2410030": "command_injection",
    "2410031": "command_injection",
    "100413": "sql_injection",
    "100420": "sql_injection",
    "100421": "sql_injection",
    "2410010": "sql_injection",
    "31164": "sql_injection",
    "100414": "xss",
    "100422": "xss",
    "2410020": "xss",
    "100412": "bruteforce",
    "100424": "bruteforce",
    "2410040": "bruteforce",
    "100410": "dos",
    "2410070": "dos",
    "100411": "ssh_bruteforce",
    "2410080": "ssh_bruteforce",
}

SIGNATURE_ATTACK_TYPES = {
    "1:2410030:1": "command_injection",
    "1:2410031:1": "command_injection",
    "1:2410010:1": "sql_injection",
    "1:2410020:1": "xss",
    "1:2410040:1": "bruteforce",
    "1:2410070:1": "dos",
    "1:2410080:1": "ssh_bruteforce",
}


def _get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _first(*values: Any, default: Any = "") -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).lower() for v in value if v is not None]
    return [str(value).lower()]


def parse_apache_full_log(full_log: str) -> Dict[str, Any]:
    match = APACHE_RE.search(str(full_log or ""))
    if not match:
        return {}
    size_text = match.group("size")
    return {
        "method": match.group("method").upper(),
        "url": match.group("url"),
        "status": _to_int(match.group("status")),
        "user_agent": match.group("ua") or "",
        "bytes_toclient": 0 if size_text == "-" else _to_int(size_text),
    }


def parse_snort_full_log(full_log: str) -> Dict[str, Any]:
    match = SNORT_RE.search(str(full_log or ""))
    if not match:
        return {}
    return {
        "signature_id": match.group("signature_id"),
        "signature": match.group("signature").strip(),
    }


def raw_http(alert: Dict[str, Any]) -> Dict[str, Any]:
    """Return normalized HTTP fields from normalized or raw Wazuh alerts."""
    http = alert.get("http") if isinstance(alert.get("http"), dict) else {}
    data_http = _get(alert, "data.http", {})
    if not isinstance(data_http, dict):
        data_http = {}
    apache = parse_apache_full_log(str(alert.get("full_log", "")))

    method = _first(
        http.get("method"),
        http.get("http_method"),
        data_http.get("method"),
        data_http.get("http_method"),
        _get(alert, "data.protocol"),
        apache.get("method"),
        default="unknown",
    )
    status = _first(
        http.get("status"),
        data_http.get("status"),
        _get(alert, "data.id"),
        apache.get("status"),
        default=0,
    )
    url = _first(
        http.get("url"),
        http.get("raw_url"),
        data_http.get("url"),
        data_http.get("raw_url"),
        _get(alert, "data.url"),
        apache.get("url"),
        default="",
    )
    user_agent = _first(
        http.get("user_agent"),
        http.get("http_user_agent"),
        data_http.get("user_agent"),
        data_http.get("http_user_agent"),
        apache.get("user_agent"),
        default="",
    )

    return {
        "method": str(method or "unknown"),
        "status": _to_int(status),
        "url": str(url or ""),
        "raw_url": str(http.get("raw_url") or url or ""),
        "user_agent": str(user_agent or ""),
        "hostname": str(_first(http.get("hostname"), data_http.get("hostname"), default="")),
        "protocol": str(_first(http.get("protocol"), data_http.get("protocol"), default="")),
    }


def raw_suricata_alert(alert: Dict[str, Any]) -> Dict[str, Any]:
    suricata = alert.get("suricata_alert") if isinstance(alert.get("suricata_alert"), dict) else {}
    data_alert = _get(alert, "data.alert", {})
    if not isinstance(data_alert, dict):
        data_alert = {}
    snort = parse_snort_full_log(str(alert.get("full_log", "")))
    return {
        "severity": _first(suricata.get("severity"), data_alert.get("severity"), default=0),
        "action": _first(suricata.get("action"), data_alert.get("action"), default="unknown"),
        "signature": _first(suricata.get("signature"), data_alert.get("signature"), snort.get("signature"), default=""),
        "category": _first(suricata.get("category"), data_alert.get("category"), default=""),
        "signature_id": _first(
            suricata.get("signature_id"),
            data_alert.get("signature_id"),
            _get(alert, "data.id"),
            snort.get("signature_id"),
            default="",
        ),
    }


def raw_flow(alert: Dict[str, Any]) -> Dict[str, Any]:
    flow = alert.get("flow") if isinstance(alert.get("flow"), dict) else {}
    data_flow = _get(alert, "data.flow", {})
    if not isinstance(data_flow, dict):
        data_flow = {}
    return {
        "bytes_toclient": _first(flow.get("bytes_toclient"), data_flow.get("bytes_toclient"), default=0),
        "bytes_toserver": _first(flow.get("bytes_toserver"), data_flow.get("bytes_toserver"), default=0),
        "src_port": _first(flow.get("src_port"), data_flow.get("src_port"), _get(alert, "data.src_port"), _get(alert, "data.srcport"), default=0),
        "dest_port": _first(flow.get("dest_port"), data_flow.get("dest_port"), _get(alert, "data.dest_port"), _get(alert, "data.dstport"), default=0),
    }


def is_internal_ip(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(str(ip))
        return bool(addr.is_private or addr.is_loopback)
    except ValueError:
        return False


def normalize_attack_type(alert: Dict[str, Any]) -> str:
    explicit = (
        alert.get("attack_type")
        or alert.get("attack_type_normalized")
        or _get(alert, "ml_metadata.attack_type")
    )
    if explicit:
        return str(explicit).lower().replace("-", "_")

    rule_id = str(_get(alert, "rule.id", ""))
    if rule_id in RAW_RULE_ATTACK_TYPES:
        return RAW_RULE_ATTACK_TYPES[rule_id]

    http = raw_http(alert)
    suricata = raw_suricata_alert(alert)
    signature_id = str(suricata.get("signature_id") or "")
    if signature_id in SIGNATURE_ATTACK_TYPES:
        return SIGNATURE_ATTACK_TYPES[signature_id]

    groups = " ".join(_as_list(_get(alert, "rule.groups", [])) + _as_list(alert.get("tags", [])))
    text = " ".join(
        [
            str(alert.get("message", "")),
            str(alert.get("full_log", "")),
            str(_get(alert, "rule.description", "")),
            str(http.get("url", "")),
            str(http.get("raw_url", "")),
            str(suricata.get("signature", "")),
            str(_get(alert, "data.payload_printable", "")),
            groups,
        ]
    ).lower()
    decoded = unquote_plus(text)

    for attack_type, patterns in ATTACK_KEYWORDS.items():
        if any(p in decoded for p in patterns):
            return attack_type
    return "normal"


def extract_features(alert: Dict[str, Any]) -> Dict[str, Any]:
    rule_level = _to_int(_get(alert, "rule.level"))
    rule_id = str(_get(alert, "rule.id", "unknown"))
    groups = _as_list(_get(alert, "rule.groups", []))
    tags = [tag for tag in _as_list(alert.get("tags", [])) if tag not in {"p1", "p2", "p3", "p4"}]
    http = raw_http(alert)
    suricata = raw_suricata_alert(alert)
    flow = raw_flow(alert)
    correlation = alert.get("correlation") or {}
    fp = alert.get("fp_filtering") or {}

    srcip = _first(
        alert.get("srcip"),
        alert.get("src_ip"),
        _get(alert, "data.srcip"),
        _get(alert, "data.src_ip"),
        _get(alert, "data.audit.srcip"),
        default="",
    )
    dest_ip = _first(
        alert.get("dest_ip"),
        alert.get("dstip"),
        _get(alert, "data.dest_ip"),
        _get(alert, "data.dstip"),
        _get(alert, "data.flow.dest_ip"),
        default="",
    )
    user_agent = str(http.get("user_agent") or http.get("http_user_agent") or "").lower()
    url = unquote_plus(str(http.get("url") or "")).lower()
    message = str(alert.get("message") or alert.get("full_log") or "").lower()
    attack_type = normalize_attack_type(alert)

    suspicious_ua = any(tool in user_agent for tool in ["sqlmap", "nikto", "nmap", "burp", "acunetix", "metasploit"])
    attack_text = " ".join([url, message, user_agent])
    attack_pattern_hits = sum(1 for patterns in ATTACK_KEYWORDS.values() for p in patterns if p in attack_text)
    has_http = bool(url or _to_int(http.get("status")) or str(http.get("method") or "").lower() != "unknown")
    has_suricata = bool(
        suricata.get("signature")
        or suricata.get("category")
        or suricata.get("signature_id")
        or _to_int(suricata.get("severity"))
    )

    features: Dict[str, Any] = {
        "rule_level": rule_level,
        "rule_id": rule_id,
        "rule_firedtimes": _to_int(_get(alert, "rule.firedtimes"), 1),
        "agent_id": str(_get(alert, "agent.id", "unknown")),
        "agent_name": str(_get(alert, "agent.name", "unknown")).lower(),
        "src_internal": int(is_internal_ip(str(srcip))),
        "dest_internal": int(is_internal_ip(str(dest_ip))),
        "has_srcip": int(bool(srcip)),
        "has_dest_ip": int(bool(dest_ip)),
        "http_method": str(http.get("method") or "unknown").upper(),
        "http_status": _to_int(http.get("status")),
        "http_status_family": str(_to_int(http.get("status")) // 100) + "xx" if http.get("status") else "unknown",
        "has_http": int(has_http),
        "suspicious_user_agent": int(suspicious_ua),
        "url_len": len(url),
        "attack_pattern_hits": attack_pattern_hits,
        "suricata_severity": _to_int(suricata.get("severity")),
        "suricata_action": str(suricata.get("action") or "unknown").lower(),
        "has_suricata": int(has_suricata),
        "bytes_toclient": _to_float(flow.get("bytes_toclient")),
        "bytes_toserver": _to_float(flow.get("bytes_toserver")),
        "correlated": int(bool(correlation.get("is_correlated"))),
        "correlation_size": _to_int(correlation.get("group_size"), 1),
        "fp_risk": str(fp.get("fp_risk") or "unknown").lower(),
        "attack_type": attack_type,
        "attack_priority": ATTACK_PRIORITY.get(attack_type, 0),
    }

    for group in groups[:12]:
        features[f"group={group}"] = 1
    for tag in tags[:12]:
        features[f"tag={tag}"] = 1

    return features


def derive_priority_label(alert: Dict[str, Any]) -> str:
    explicit = alert.get("priority") or alert.get("label")
    if explicit in {"P1", "P2", "P3", "P4"}:
        return str(explicit)

    level = _to_int(_get(alert, "rule.level"))
    attack_type = normalize_attack_type(alert)
    is_attack = alert.get("is_attack")

    if attack_type in {"command_injection", "webshell", "sql_injection", "dos"}:
        return "P1"
    if attack_type in {"xss", "ssh_bruteforce"}:
        return "P2"
    if attack_type in {"bruteforce", "malware_tool"}:
        return "P3"
    if level >= 12:
        return "P1"
    if level >= 10:
        return "P2"
    if level >= 7:
        return "P3"
    if is_attack is True:
        return "P3"
    groups = " ".join(_as_list(_get(alert, "rule.groups", [])))
    desc = str(_get(alert, "rule.description", "")).lower()
    if any(x in groups for x in ["attack", "authentication_failed", "ids"]):
        return "P3"
    if any(x in desc for x in ["failed", "invalid", "denied"]):
        return "P3"
    return "P4"


def extract_many(alerts: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [extract_features(a) for a in alerts]
