"""
Flask service  Extract static features PE file theo format EMBER2024

"""

import array
import math
import os
import hashlib
import json
import socket
import sys
import threading
import time
from pathlib import Path

import pefile
import requests
from flask import Flask, jsonify, request

app = Flask(__name__)


# Nhiem vu chinh cua file:
#   1. Nhan duong dan file PE (.exe/.dll) tu API /analyze hoac tu auto-scan.
#   2. Doc byte cua file do.
#   3. Trich xuat feature theo format EMBER2024 bang PEFeatureExtractor.
#   4. Tao vector 2560 so thuc.
#   5. Gui vector nay bang HTTP POST sang combined_ml_server.py endpoint
#      /predict-features de combined server du doan benign/malicious.
#
# Tu khoa de tim nhanh trong file:
#   - "extract_infos"              : noi doc file va trich xuat feature vector.
#   - "get_thrember_extractor"     : noi tao/cau hinh PEFeatureExtractor.
#   - "requests.post"              : noi gui POST sang combined ML server.
#   - "analyze_file"               : ham dung cho auto scan/monitor.
#   - "@app.route(\"/analyze\")"    : API Flask nhan request tu ben ngoai.
#   - "scan_directory"             : quet thu muc Downloads khi service start.
#   - "monitor_directory"          : theo doi Downloads lien tuc.
# =============================================================================

# Dia chi combined_ml_server.py.
# Neu combined chay local thi thuong la:
#   http://127.0.0.1:8000/predict-features
# hoac:
#   http://127.0.0.1:8010/predict-features
#
# Co the override ma khong can sua code bang bien moi truong:
#   set ML_SERVER_URL=http://127.0.0.1:8000/predict-features
ML_SERVER_URL = os.getenv("ML_SERVER_URL", "http://127.0.0.1:8000/predict-features")

# Cac thu muc se duoc auto-scan neu khong set PE_SCAN_DIR.

DEFAULT_SCAN_DIR_CANDIDATES = [
    r"C:\User\sangd\Downloads",
    r"C:\Users\sangd\Downloads",
]
# PE_AUTO_SCAN_ON_START=1: khi service start, quet Downloads mot lan.
# PE_WATCH_SCAN_DIR=1   : sau do tiep tuc theo doi Downloads lien tuc.
AUTO_SCAN_ON_START = os.getenv("PE_AUTO_SCAN_ON_START", "1").lower() not in ("0", "false", "no")
WATCH_SCAN_DIR = os.getenv("PE_WATCH_SCAN_DIR", "1").lower() not in ("0", "false", "no")
WATCH_INTERVAL_SECONDS = int(os.getenv("PE_WATCH_INTERVAL_SECONDS", "10"))


MODEL_META = {
    "model_path":    r"E:\ML Service\ember2024_pe_lgbm.model",
    "num_trees":     999,
    # combined_ml_server.py dang can dung 2560 features.
    # PEFeatureExtractor goc co 2568 features; ben duoi ta bo nhom
    # "authenticode" 8 chieu de con dung 2560.
    "num_features":  2560,
    "auc_roc":       0.9983,
    "fpr":           0.0123,
    "fnr":           0.0264,
}

# Categorical feature indices
# index 2   = is_pe
# index 3-6 = start_bytes[0-3]
# index 701 = machine_type
# index 702 = subsystem
CATEGORICAL_FEATURE_INDICES = [2, 3, 4, 5, 6, 701, 702]


"""
PHAN EMBEDDED EMBER2024 FEATURE EXTRACTOR
-----------------------------------------
Tu doan nay tro xuong la code trich xuat feature duoc nhung truc tiep tu
EMBER2024/thrember/features.py. Muc dich nhung truc tiep la de file service nay
co the tu chay doc lap, khong can import features.py tu thu muc EMBER2024 nua.

Neu ban chi can tim noi "service lay feature tu file PE nhu the nao", hay tim
ham extract_infos() o gan cuoi phan extractor.


"""

import os
import hashlib
import json
import re
import io
import math
from pathlib import Path
from collections import Counter, OrderedDict

import numpy as np
import pefile
from sklearn.feature_extraction import FeatureHasher
try:
    import signify
    from signify.authenticode import SignedPEFile
except ImportError:
    signify = None
    SignedPEFile = None
from datetime import datetime


class FeatureType(object):
    """
    Base class from which each feature type may inherit
    """

    name: str = ""
    dim: int = 0

    def __repr__(self):
        return "{}({})".format(self.name, self.dim)

    def raw_features(self, bytez: bytes, pe: pefile.PE | None = None):
        """Generate a JSON-able representation of the file"""
        raise (NotImplementedError)

    def process_raw_features(self, raw_obj):
        """Generate a feature vector from the raw features"""
        raise (NotImplementedError)

    def feature_vector(self, bytez: bytes, pe: pefile.PE | None = None):
        """Directly calculate the feature vector from the sample itself. This should only be implemented differently
        if there are significant speedups to be gained from combining the two functions."""
        return self.process_raw_features(self.raw_features(bytez, pe))


class GeneralFileInfo(FeatureType):
    """
    General information about the file
    """

    name = "general"
    dim = 3 + 4

    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, bytez, pe=None):
        # From pefile entropy_H()
        size = len(bytez)
        bytez_arr = bytearray(bytez)
        occurences = Counter(bytez_arr)
        entropy = 0
        for x in occurences.values():
            p_x = float(x) / size
            entropy -= p_x * math.log(p_x, 2)

        raw_obj = {
            "size": size,
            "entropy": entropy,
            "is_pe": 0 if pe is None else 1,
            "start_bytes": [
                int(bytez_arr[0]),
                int(bytez_arr[1]) if size >= 2 else 0,
                int(bytez_arr[2]) if size >= 3 else 0,
                int(bytez_arr[3]) if size >= 4 else 0,
            ],
        }
        return raw_obj

    def process_raw_features(self, raw_obj):
        return np.hstack(
            [
                raw_obj["size"],
                raw_obj["entropy"],
                raw_obj["is_pe"], 
                raw_obj["start_bytes"], 
            ],
            dtype=np.float32,
        )


class ByteHistogram(FeatureType):
    """
    Byte histogram (count + non-normalized) over the entire binary file
    """

    name = "histogram"
    dim = 256

    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, bytez, pe):
        counts = np.bincount(np.frombuffer(bytez, dtype=np.uint8), minlength=256)
        return counts.tolist()
    
    def process_raw_features(self, raw_obj):
        counts = np.array(raw_obj, dtype=np.float32)
        sum = counts.sum()
        normalized = counts / sum
        return normalized


class ByteEntropyHistogram(FeatureType):


    name = "byteentropy"
    dim = 256

    def __init__(self, step=1024, window=2048):
        super(FeatureType, self).__init__()
        self.window = window
        self.step = step

    def _entropy_bin_counts(self, block):
        
        c = np.bincount(block >> 4, minlength=16)  #
        p = c.astype(np.float32) / self.window
        wh = np.where(c)[0]
       
        H = np.sum(-p[wh] * np.log2(p[wh])) * 2

        Hbin = int(H * 2)  
        if Hbin == 16:  
            Hbin = 15

        return Hbin, c

    def raw_features(self, bytez, pe):
        output = np.zeros((16, 16), dtype=np.int32)
        a = np.frombuffer(bytez, dtype=np.uint8)
        if a.shape[0] < self.window:
            Hbin, c = self._entropy_bin_counts(a)
            output[Hbin, :] += c
        else:
            
            shape = a.shape[:-1] + (a.shape[-1] - self.window + 1, self.window)
            strides = a.strides + (a.strides[-1],)
            blocks = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[:: self.step, :]

            
            for block in blocks:
                Hbin, c = self._entropy_bin_counts(block)
                output[Hbin, :] += c

        return output.flatten().tolist()

    def process_raw_features(self, raw_obj):
        counts = np.array(raw_obj, dtype=np.float32)
        sum = counts.sum()
        normalized = counts / sum
        return normalized


class StringExtractor(FeatureType):


    name = "strings"
    dim = 5 + 96 + 76

    def __init__(self):
        super(FeatureType, self).__init__()
        
        self._allstrings = re.compile(b"[\x20-\x7f]{5,}")

        
        self._regexes = {
            
            "url": re.compile("\\b(?:http|https|ftp):\\/\\/[a-zA-Z0-9-._~:?#[\\]@!$&'()*+,;=]+"),
            "ipv4_addr": re.compile("\\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\\b"),
            "ipv6_addr": re.compile("\\b(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}\\b|\\b(?:[A-Fa-f0-9]{1,4}:){1,7}:\\b|\\b:[A-Fa-f0-9]{1,4}(?::[A-Fa-f0-9]{1,4}){1,6}\\b"),
            "mac_addr": re.compile("\\b(?:[0-9A-Fa-f]{2}[:-]){5}(?:[0-9A-Fa-f]{2})\\b"),
            "email_addr": re.compile("\\b(?:[0-9A-Fa-f]{2}[:-]){5}(?:[0-9A-Fa-f]{2})\\b"),
            "btc_wallet": re.compile("[13][a-km-zA-HJ-NP-Z1-9]{25,34}"),

            
            "file_path": re.compile("\\bC:/"),
            "dos_msg": re.compile("!This program "),
            "registry_key": re.compile("\\b(?:KHEY_|KHLM|HKCU)"),

            
            "/dev/": re.compile("/dev/"),
            "/proc/": re.compile("/proc/"),
            "/bin/": re.compile("/bin/"),
            "/usr/": re.compile("/usr/"),
            "/tmp/": re.compile("/tmp/"),

            
            "/URI": re.compile("/URI"),
            "/FlateDecode": re.compile("/FlateDecode"),
            "/EmbeddedFile": re.compile("/EmbeddedFile"),

            
            "html": re.compile("html", re.IGNORECASE),
            "javascript": re.compile("javascript", re.IGNORECASE),
            "<script": re.compile("<script", re.IGNORECASE),
            ".click(": re.compile(".click", re.IGNORECASE),
            "onlick": re.compile("onclick", re.IGNORECASE),

            
            "powershell": re.compile("powershell", re.IGNORECASE),
            "Invoke-Expression": re.compile("Invoke-Expression"),
            "Invoke-Command": re.compile("Invoke-Command"),
            "Start-process": re.compile("Start-process"),

            
            "get": re.compile("GET /", re.IGNORECASE),
            "post": re.compile("POST /", re.IGNORECASE),
            "http": re.compile("HTTP/", re.IGNORECASE),
            "http://": re.compile("http://", re.IGNORECASE),
            "https://": re.compile("https://", re.IGNORECASE),
            "ftp": re.compile("ftp:", re.IGNORECASE),
            "useragent": re.compile("User-Agent", re.IGNORECASE),
            "cookie": re.compile("cookie", re.IGNORECASE),
            "internet": re.compile("internet", re.IGNORECASE),
            "download": re.compile("download", re.IGNORECASE),
            "connect": re.compile("connect", re.IGNORECASE),

            
            "base64": re.compile("base64", re.IGNORECASE),
            "base64string": re.compile("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"),
            "crypt": re.compile("crypt"),
            "encode": re.compile("encode", re.IGNORECASE),
            "decode": re.compile("decode", re.IGNORECASE),

            
            "cache": re.compile("cache", re.IGNORECASE),
            "certificate": re.compile("certificate", re.IGNORECASE),
            "clipboard": re.compile("clipboard", re.IGNORECASE),
            "command": re.compile("command", re.IGNORECASE),
            "create": re.compile("create", re.IGNORECASE),
            "debug": re.compile("debug", re.IGNORECASE),
            "delete": re.compile("delete", re.IGNORECASE),
            "desktop": re.compile("desktop", re.IGNORECASE),
            "directory": re.compile("directory", re.IGNORECASE),
            "disk": re.compile("disk", re.IGNORECASE),
            "environment": re.compile("environment", re.IGNORECASE),
            "enum": re.compile("enum", re.IGNORECASE),
            "exit": re.compile("exit", re.IGNORECASE),
            "file": re.compile("file", re.IGNORECASE),
            "hostname": re.compile("hostname", re.IGNORECASE),
            "install": re.compile("install", re.IGNORECASE),
            "hidden": re.compile("hidden", re.IGNORECASE),
            "keyboard": re.compile("keyboard", re.IGNORECASE),
            "memory": re.compile("memory", re.IGNORECASE),
            "module": re.compile("module", re.IGNORECASE),
            "mutex": re.compile("mutex", re.IGNORECASE),
            "password": re.compile("password", re.IGNORECASE),
            "privilege": re.compile("privilege", re.IGNORECASE),
            "process": re.compile("process", re.IGNORECASE),
            "remote": re.compile("remote", re.IGNORECASE),
            "resource": re.compile("resource", re.IGNORECASE),
            "security": re.compile("security", re.IGNORECASE),
            "service": re.compile("service", re.IGNORECASE),
            "shell": re.compile("shell", re.IGNORECASE),
            "snapshot": re.compile("snapshot", re.IGNORECASE),
            "system": re.compile("system", re.IGNORECASE),
            "thread": re.compile("thread", re.IGNORECASE),
            "token": re.compile("token", re.IGNORECASE),
            "wallet": re.compile("wallet", re.IGNORECASE),
            "window": re.compile("window", re.IGNORECASE),
        }
        self.regex_idxs = {k: v for v, k in enumerate(sorted(self._regexes))}

    def raw_features(self, bytez, pe):
        allstrings = self._allstrings.findall(bytez)
        allstrings_ascii = [s.decode() for s in allstrings]
        if allstrings:
            # Tinh cac thong ke ve chuoi ASCII tim thay trong file.
            string_lengths = [len(s) for s in allstrings]
            avlength = sum(string_lengths) / len(string_lengths)
            
            as_shifted_string = [b - ord(b"\x20") for b in b"".join(allstrings)]
            c = np.bincount(as_shifted_string, minlength=96)  
            
            csum = c.sum()
            p = c.astype(np.float32) / csum
            wh = np.where(c)[0]
            H = np.sum(-p[wh] * np.log2(p[wh]))  

        else:
            avlength = 0
            c = np.zeros((96,), dtype=np.float32)
            H = 0
            csum = 0

        
        string_counts = {}
        for s in allstrings_ascii:
            for k, r in self._regexes.items():
                if re.search(r, s):
                    if string_counts.get(k) is None:
                        string_counts[k] = 0
                    string_counts[k] += 1
        string_counts = OrderedDict(sorted(string_counts.items()))

        return {
            "numstrings": len(allstrings),
            "avlength": avlength,
            "printabledist": c.tolist(),  
            "printables": int(csum),
            "entropy": float(H),
            "string_counts": string_counts,
        }

    def process_raw_features(self, raw_obj):
        hist_divisor = float(raw_obj["printables"]) if raw_obj["printables"] > 0 else 1.0
        string_counts = np.zeros(len(self.regex_idxs), dtype=np.float32)
        for regex, count in raw_obj["string_counts"].items():
            idx = self.regex_idxs[regex]
            string_counts[idx] = count

        return np.hstack(
            [
                raw_obj["numstrings"],
                raw_obj["avlength"],
                raw_obj["printables"],
                np.asarray(raw_obj["printabledist"]) / hist_divisor,
                raw_obj["entropy"],
                string_counts,
            ]
        ).astype(np.float32)


class SectionInfo(FeatureType):


    name = "section"
    dim = 11 + 50 + 50 + 50 + 50 + 10 + 3

    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, bytez, pe):
        if pe is None:
            return {}

       
        entry_section = ""
        aoep = pe.OPTIONAL_HEADER.AddressOfEntryPoint
        for section in pe.sections:
            if section.contains_rva(aoep):
                entry_section = section.Name.strip(b"\x00").decode(errors="ignore").lower()

        isection = 0
        while entry_section == "" and isection < len(pe.sections):
            if pe.sections[isection].Characteristics & 0x20000000 > 0:
                entry_section = pe.sections[isection].Name.strip(b"\x00").decode(errors="ignore").lower()
            isection += 1

        raw_obj = {"entry": entry_section}
        raw_obj["sections"] = [
            {
                "name": section.Name.strip(b"\x00").decode(errors="ignore").lower(),
                "size": section.SizeOfRawData,
                "entropy": section.get_entropy(),
                "vsize": section.Misc_VirtualSize,
                "size_ratio": section.SizeOfRawData / len(bytez),
                "vsize_ratio": section.SizeOfRawData / max(section.Misc_VirtualSize, 1),
                "props": [sc[10:] for sc, _ in pefile.section_characteristics if section.__dict__[sc]],
            }
            for section in pe.sections
        ]
        raw_obj["overlay"] = {
            "size": 0,
            "size_ratio": 0,
            "entropy": 0,
        }

        overlay = pe.get_overlay()
        if overlay is not None:

            
            overlay_size = len(overlay)
            occurences = Counter(bytearray(overlay))
            entropy = 0
            for x in occurences.values():
                p_x = float(x) / len(overlay)
                entropy -= p_x * math.log(p_x, 2)

            raw_obj["overlay"] = {
                "size": overlay_size,
                "size_ratio": overlay_size / len(bytez),
                "entropy": entropy
            }

        return raw_obj

    def process_raw_features(self, raw_obj):
        if not raw_obj:
            return np.zeros(self.dim, dtype=np.float32)

        sections = raw_obj["sections"]

       
        n_sections = len(sections)
        n_zero_size = sum(1 for s in sections if s["size"] == 0)
        n_emtpy_name = sum(1 for s in sections if s["name"] == "")
        n_rx = sum(1 for s in sections if "MEM_READ" in s["props"] and "MEM_EXECUTE" in s["props"])
        n_w = sum(1 for s in sections if "MEM_WRITE" in s["props"])
        entropies = [s["entropy"] for s in sections] + [raw_obj["overlay"]["entropy"]] + [0]
        size_ratios = [s["size_ratio"] for s in sections] + [raw_obj["overlay"]["size_ratio"]] + [0]
        vsize_ratios = [s["vsize_ratio"] for s in sections] + [0]

        general = [
            n_sections,
            n_zero_size,
            n_emtpy_name,
            n_rx,
            n_w,
            max(entropies),
            min(entropies),
            max(size_ratios),
            min(size_ratios),
            max(vsize_ratios),
            min(vsize_ratios)
        ]

        
        section_sizes = [(s["name"], s["size"]) for s in sections]
        section_sizes_hashed = FeatureHasher(50, input_type="pair").transform([section_sizes]).toarray()[0]
        section_vsize = [(s["name"], s["vsize"]) for s in sections]
        section_vsize_hashed = FeatureHasher(50, input_type="pair").transform([section_vsize]).toarray()[0]
        section_entropy = [(s["name"], s["entropy"]) for s in sections]
        section_entropy_hashed = FeatureHasher(50, input_type="pair").transform([section_entropy]).toarray()[0]
        characteristics = [f"{s['name']}:{p}" for s in sections for p in s["props"]]
        characteristics_hashed = FeatureHasher(50, input_type="string").transform([characteristics]).toarray()[0]
        entry_name_hashed = FeatureHasher(10, input_type="string").transform([[raw_obj["entry"]]]).toarray()[0]

        return np.hstack(
            [
                general,
                section_sizes_hashed,
                section_vsize_hashed,
                section_entropy_hashed,
                characteristics_hashed,
                entry_name_hashed,
                raw_obj["overlay"]["size"],
                raw_obj["overlay"]["size_ratio"],
                raw_obj["overlay"]["entropy"]
            ]
        ).astype(np.float32)


class ImportsInfo(FeatureType):


    name = "imports"
    dim = 2 + 256 + 1024

    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, bytez, pe):
        imports = {}
        if pe is None or "DIRECTORY_ENTRY_IMPORT" not in pe.__dict__.keys():
            return imports

        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            dll_name = entry.dll.decode()
            imports[dll_name] = []

            
            for lib in entry.imports:
                if lib.name is not None and len(lib.name):
                    imports[dll_name].append(lib.name.decode()[:10000])
                elif lib.ordinal is not None:
                    imports[dll_name].append(f"{dll_name}:ordinal{lib.ordinal}")

        return imports

    def process_raw_features(self, raw_obj):
        if not raw_obj:
            return np.zeros(self.dim, dtype=np.float32)

        
        libraries = list(set([l.lower() for l in raw_obj.keys()]))
        libraries_hashed = FeatureHasher(256, input_type="string", alternate_sign=False).transform([libraries]).toarray()[0]

       
        imports = [lib.lower() + ":" + e for lib, elist in raw_obj.items() for e in elist]
        imports_hashed = FeatureHasher(1024, input_type="string", alternate_sign=False).transform([imports]).toarray()[0]

        lengths = [len(imports), len(libraries)]

        
        return np.hstack([lengths, libraries_hashed, imports_hashed]).astype(np.float32)


class ExportsInfo(FeatureType):


    name = "exports"
    dim = 1 + 128

    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, bytez, pe):
        if pe is None:
            return []

        clipped_exports = []
        if "DIRECTORY_ENTRY_EXPORT" in pe.__dict__.keys():
            for exp in pe.DIRECTORY_ENTRY_EXPORT.symbols:
                if exp.name is not None and len(exp.name):
                    clipped_exports.append(exp.name.decode()[:10000])
                elif exp.ordinal is not None:
                    clipped_exports.append(f"ordinal{exp.ordinal}")

        return clipped_exports

    def process_raw_features(self, raw_obj):
        if not raw_obj:
            return np.zeros(self.dim, dtype=np.float32)

        exports_hashed = FeatureHasher(128, input_type="string").transform([raw_obj]).toarray()[0]
        return np.hstack([np.array([len(exports_hashed)]), exports_hashed.astype(np.float32)])


class HeaderFileInfo(FeatureType):


    name = "header"
    dim = 74

    def __init__(self):
        super(FeatureType, self).__init__()
       
        self._machine_types = [
            "IMAGE_FILE_MACHINE_UNKNOWN",
            "IMAGE_FILE_MACHINE_I386",
            "IMAGE_FILE_MACHINE_R3000",
            "IMAGE_FILE_MACHINE_R4000",
            "IMAGE_FILE_MACHINE_R10000",
            "IMAGE_FILE_MACHINE_WCEMIPSV2",
            "IMAGE_FILE_MACHINE_ALPHA",
            "IMAGE_FILE_MACHINE_SH3",
            "IMAGE_FILE_MACHINE_SH3DSP",
            "IMAGE_FILE_MACHINE_SH3E",
            "IMAGE_FILE_MACHINE_SH4",
            "IMAGE_FILE_MACHINE_SH5",
            "IMAGE_FILE_MACHINE_ARM",
            "IMAGE_FILE_MACHINE_THUMB",
            "IMAGE_FILE_MACHINE_ARMNT",
            "IMAGE_FILE_MACHINE_AM33",
            "IMAGE_FILE_MACHINE_POWERPC",
            "IMAGE_FILE_MACHINE_POWERPCFP",
            "IMAGE_FILE_MACHINE_IA64",
            "IMAGE_FILE_MACHINE_MIPS16",
            "IMAGE_FILE_MACHINE_ALPHA64",
            "IMAGE_FILE_MACHINE_AXP64",
            "IMAGE_FILE_MACHINE_MIPSFPU",
            "IMAGE_FILE_MACHINE_MIPSFPU16",
            "IMAGE_FILE_MACHINE_TRICORE",
            "IMAGE_FILE_MACHINE_CEF",
            "IMAGE_FILE_MACHINE_EBC",
            "IMAGE_FILE_MACHINE_RISCV32",
            "IMAGE_FILE_MACHINE_RISCV64",
            "IMAGE_FILE_MACHINE_RISCV128",
            "IMAGE_FILE_MACHINE_LOONGARCH32",
            "IMAGE_FILE_MACHINE_LOONGARCH64",
            "IMAGE_FILE_MACHINE_AMD64",
            "IMAGE_FILE_MACHINE_M32R",
            "IMAGE_FILE_MACHINE_ARM64",
            "IMAGE_FILE_MACHINE_CEE",
        ]
        self._machine_types_dict = dict([(mt, i) for i, mt in enumerate(self._machine_types)])
        self._subsystem_types = [
            "IMAGE_SUBSYSTEM_UNKNOWN",
            "IMAGE_SUBSYSTEM_NATIVE",
            "IMAGE_SUBSYSTEM_WINDOWS_GUI",
            "IMAGE_SUBSYSTEM_WINDOWS_CUI",
            "IMAGE_SUBSYSTEM_OS2_CUI",
            "IMAGE_SUBSYSTEM_POSIX_CUI",
            "IMAGE_SUBSYSTEM_NATIVE_WINDOWS",
            "IMAGE_SUBSYSTEM_WINDOWS_CE_GUI",
            "IMAGE_SUBSYSTEM_EFI_APPLICATION",
            "IMAGE_SUBSYSTEM_EFI_BOOT_SERVICE_DRIVER",
            "IMAGE_SUBSYSTEM_EFI_RUNTIME_DRIVER",
            "IMAGE_SUBSYSTEM_EFI_ROM",
            "IMAGE_SUBSYSTEM_XBOX",
            "IMAGE_SUBSYSTEM_WINDOWS_BOOT_APPLICATION",
        ]
        self._subsystem_types_dict = dict([(st, i) for i, st in enumerate(self._subsystem_types)])
        self._image_characteristics = [
            "RELOCS_STRIPPED",
            "EXECUTABLE_IMAGE",
            "LINE_NUMS_STRIPPED",
            "LOCAL_SYMS_STRIPPED",
            "AGGRESIVE_WS_TRIM",
            "LARGE_ADDRESS_AWARE",
            "16BIT_MACHINE",
            "BYTES_REVERSED_LO",
            "32BIT_MACHINE",
            "DEBUG_STRIPPED",
            "REMOVABLE_RUN_FROM_SWAP",
            "NET_RUN_FROM_SWAP",
            "SYSTEM",
            "DLL",
            "UP_SYSTEM_ONLY",
            "BYTES_REVERSED_HI",
        ]
        self._dll_characteristics = [
            "HIGH_ENTROPY_VA",
            "DYNAMIC_BASE",
            "FORCE_INTEGRITY",
            "NX_COMPAT",
            "NO_ISOLATION",
            "NO_SEH",
            "NO_BIND",
            "APPCONTAINER",
            "WDM_DRIVER",
            "GUARD_CF",
            "TERMINAL_SERVER_AWARE",
        ]
        self._dos_members = [
            "e_magic",
            "e_cblp",
            "e_cp",
            "e_crlc",
            "e_cparhdr",
            "e_minalloc",
            "e_maxalloc",
            "e_ss",
            "e_sp",
            "e_csum",
            "e_ip",
            "e_cs",
            "e_lfarlc",
            "e_ovno",
            "e_oemid",
            "e_oeminfo",
            "e_lfanew",
        ]

    def raw_features(self, bytez, pe):
        if pe is None:
            return {}

        raw_obj = {}
        raw_obj["coff"] = {
            "timestamp": 0,
            "machine": "",
            "number_of_sections": 0,
            "number_of_symbols": 0,
            "sizeof_optional_header": 0,
            "pointer_to_symbol_table": 0,
            "characteristics": [],
        }
        raw_obj["optional"] = {
            "magic": 0,
            "subsystem": "",
            "major_image_version": 0,
            "minor_image_version": 0,
            "major_linker_version": 0,
            "minor_linker_version": 0,
            "major_operating_system_version": 0,
            "minor_operating_system_version": 0,
            "major_subsystem_version": 0,
            "minor_subsystem_version": 0,
            "sizeof_code": 0,
            "sizeof_headers": 0,
            "sizeof_image": 0,
            "sizeof_initialized_data": 0,
            "sizeof_uninitialized_data": 0,
            "sizeof_stack_reserve": 0,
            "sizeof_stack_commit": 0,
            "sizeof_heap_reserve": 0,
            "sizeof_heap_commit": 0,
            "address_of_entrypoint": 0,
            "base_of_code": 0,
            "base_of_data": 0,
            "image_base": 0,
            "section_alignment": 0,
            "checksum": 0,
            "number_of_rvas_and_sizes": 0,
            "dll_characteristics": [],
        }
        raw_obj["dos"] = {member: 0 for member in self._dos_members}
        if pe is None:
            return raw_obj

        raw_obj["coff"]["timestamp"] = pe.FILE_HEADER.TimeDateStamp
        raw_obj["coff"]["machine"] = pefile.MACHINE_TYPE.get(pe.FILE_HEADER.Machine, "IMAGE_FILE_MACHINE_UNKNOWN")
        raw_obj["coff"]["number_of_sections"] = pe.FILE_HEADER.NumberOfSections
        raw_obj["coff"]["number_of_symbols"] = pe.FILE_HEADER.NumberOfSymbols
        raw_obj["coff"]["sizeof_optional_header"] = pe.FILE_HEADER.SizeOfOptionalHeader
        raw_obj["coff"]["pointer_to_symbol_table"] = pe.FILE_HEADER.PointerToSymbolTable
        raw_obj["coff"]["characteristics"] = [
            k[11:] for k, v in pe.FILE_HEADER.__dict__.items() if k.startswith("IMAGE_FILE_") and v
        ]
        raw_obj["optional"]["magic"] = pe.OPTIONAL_HEADER.Magic
        raw_obj["optional"]["subsystem"] = pefile.SUBSYSTEM_TYPE.get(
            pe.OPTIONAL_HEADER.Subsystem, "IMAGE_SUBSYSTEM_UNKNOWN"
        )
        raw_obj["optional"]["major_image_version"] = pe.OPTIONAL_HEADER.MajorImageVersion
        raw_obj["optional"]["minor_image_version"] = pe.OPTIONAL_HEADER.MinorImageVersion
        raw_obj["optional"]["major_linker_version"] = pe.OPTIONAL_HEADER.MajorLinkerVersion
        raw_obj["optional"]["minor_linker_version"] = pe.OPTIONAL_HEADER.MinorLinkerVersion
        raw_obj["optional"]["major_operating_system_version"] = pe.OPTIONAL_HEADER.MajorOperatingSystemVersion
        raw_obj["optional"]["minor_operating_system_version"] = pe.OPTIONAL_HEADER.MinorOperatingSystemVersion
        raw_obj["optional"]["major_subsystem_version"] = pe.OPTIONAL_HEADER.MajorSubsystemVersion
        raw_obj["optional"]["minor_subsystem_version"] = pe.OPTIONAL_HEADER.MinorSubsystemVersion
        raw_obj["optional"]["sizeof_code"] = pe.OPTIONAL_HEADER.SizeOfCode
        raw_obj["optional"]["sizeof_headers"] = pe.OPTIONAL_HEADER.SizeOfHeaders
        raw_obj["optional"]["sizeof_image"] = pe.OPTIONAL_HEADER.SizeOfImage
        raw_obj["optional"]["sizeof_initialized_data"] = pe.OPTIONAL_HEADER.SizeOfInitializedData
        raw_obj["optional"]["sizeof_uninitialized_data"] = pe.OPTIONAL_HEADER.SizeOfUninitializedData
        raw_obj["optional"]["sizeof_stack_reserve"] = pe.OPTIONAL_HEADER.SizeOfStackReserve
        raw_obj["optional"]["sizeof_stack_commit"] = pe.OPTIONAL_HEADER.SizeOfStackCommit
        raw_obj["optional"]["sizeof_heap_reserve"] = pe.OPTIONAL_HEADER.SizeOfHeapReserve
        raw_obj["optional"]["sizeof_heap_commit"] = pe.OPTIONAL_HEADER.SizeOfHeapCommit
        raw_obj["optional"]["address_of_entrypoint"] = pe.OPTIONAL_HEADER.AddressOfEntryPoint
        raw_obj["optional"]["base_of_code"] = pe.OPTIONAL_HEADER.BaseOfCode
        raw_obj["optional"]["image_base"] = pe.OPTIONAL_HEADER.ImageBase
        raw_obj["optional"]["section_alignment"] =  pe.OPTIONAL_HEADER.SectionAlignment
        raw_obj["optional"]["checksum"] =  pe.OPTIONAL_HEADER.CheckSum
        raw_obj["optional"]["number_of_rvas_and_sizes"] = pe.OPTIONAL_HEADER.NumberOfRvaAndSizes
        raw_obj["optional"]["dll_characteristics"] = [
            k[25:] for k, v in pe.OPTIONAL_HEADER.__dict__.items() if k.startswith("IMAGE_DLLCHARACTERISTICS_") and v
        ]
        dos_dict = pe.DOS_HEADER.dump_dict()
        for member in self._dos_members:
            if dos_dict[member].get("Value") is not None:
                raw_obj["dos"][member] = dos_dict[member]["Value"]
        return raw_obj

    def process_raw_features(self, raw_obj):
        if not raw_obj:
            return np.zeros(self.dim, dtype=np.float32)

        return np.hstack(
            [
                raw_obj["coff"]["timestamp"],
                raw_obj["coff"]["number_of_sections"],
                raw_obj["coff"]["number_of_symbols"],
                raw_obj["coff"]["sizeof_optional_header"],
                raw_obj["coff"]["pointer_to_symbol_table"],
                self._machine_types_dict.get(raw_obj["coff"]["machine"], 0), 
                self._subsystem_types_dict.get(raw_obj["optional"]["subsystem"], 0),  
                raw_obj["optional"]["major_image_version"],
                raw_obj["optional"]["minor_image_version"],
                raw_obj["optional"]["major_linker_version"],
                raw_obj["optional"]["minor_linker_version"],
                raw_obj["optional"]["major_operating_system_version"],
                raw_obj["optional"]["minor_operating_system_version"],
                raw_obj["optional"]["major_subsystem_version"],
                raw_obj["optional"]["minor_subsystem_version"],
                raw_obj["optional"]["sizeof_code"],
                raw_obj["optional"]["sizeof_headers"],
                raw_obj["optional"]["sizeof_image"],
                raw_obj["optional"]["sizeof_initialized_data"],
                raw_obj["optional"]["sizeof_uninitialized_data"],
                raw_obj["optional"]["sizeof_stack_reserve"],
                raw_obj["optional"]["sizeof_stack_commit"],
                raw_obj["optional"]["sizeof_heap_reserve"],
                raw_obj["optional"]["sizeof_heap_commit"],
                raw_obj["optional"]["address_of_entrypoint"],
                raw_obj["optional"]["base_of_code"],
                raw_obj["optional"]["image_base"],
                raw_obj["optional"]["section_alignment"],
                raw_obj["optional"]["checksum"],
                raw_obj["optional"]["number_of_rvas_and_sizes"],
                [ch in raw_obj["coff"]["characteristics"] for ch in self._image_characteristics],
                [ch in raw_obj["optional"]["dll_characteristics"] for ch in self._dll_characteristics],
                [raw_obj["dos"][member] for member in self._dos_members],
            ]
        ).astype(np.float32)


class DataDirectories(FeatureType):


    name = "datadirectories"
    dim = 16*2 + 2

    def __init__(self):
        super(FeatureType, self).__init__()
        self._name_order = [
            "EXPORT",
            "IMPORT",
            "RESOURCE",
            "EXCEPTION",
            "SECURITY",
            "BASERELOC",
            "DEBUG",
            "COPYRIGHT",
            "GLOBALPTR",
            "TLS",
            "LOAD_CONFIG",
            "BOUND_IMPORT",
            "IAT",
            "DELAY_IMPORT",
            "COM_DESCRIPTOR",
            "RESERVED",
        ]

    def raw_features(self, bytez, pe):
        output = []
        if pe is None:
            return output

        output.append(
            {
                "has_relocs": int(pe.has_relocs()),
                "has_dynamic_relocs": int(pe.has_dynamic_relocs()),
            }
        )

        for data_directory in pe.OPTIONAL_HEADER.DATA_DIRECTORY:
            output.append(
                {
                    "name": str(data_directory.name).replace("IMAGE_DIRECTORY_ENTRY_", ""),
                    "size": data_directory.Size,
                    "virtual_address": data_directory.VirtualAddress,
                }
            )

        return output

    def process_raw_features(self, raw_obj):
        if not raw_obj:
            return np.zeros(self.dim, dtype=np.float32)

        features = np.zeros(2 * len(self._name_order) + 2, dtype=np.float32)
        for i in range(1, len(raw_obj)-1):
            idx = self._name_order.index(raw_obj[i]["name"])
            features[2 * idx] = raw_obj[i]["size"]
            features[2 * idx + 1] = raw_obj[i]["virtual_address"]
        features[-2] = raw_obj[0]["has_relocs"]
        features[-1] = raw_obj[0]["has_dynamic_relocs"]
        return features


class RichHeader(FeatureType):
    """
    Extracts features based on the file's rich header information
    """

    name = "richheader"
    dim = 1 + 32

    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, bytez, pe):
        if pe is not None and pe.RICH_HEADER is not None:
            return pe.RICH_HEADER.values
        return []

    def process_raw_features(self, raw_obj):
        if not raw_obj:
            return np.zeros(self.dim, dtype=np.float32)

        number_of_pairs = int(len(raw_obj) / 2)
        paired_values = [(str(raw_obj[i]), raw_obj[i + 1]) for i in range(0, len(raw_obj) - 1, 2)]
        paired_values_hashed = FeatureHasher(32, input_type="pair").transform([paired_values]).toarray()[0]
        return np.hstack([number_of_pairs, paired_values_hashed]).astype(np.float32)


class AuthenticodeSignature(FeatureType):
    """
    Extracts Authenticode Digital Signature features
    """

    name = "authenticode"
    dim = 8

    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, bytez, pe):
        if pe is None:
            return {}

        raw_obj = {
            "num_certs": 0,
            "self_signed": 0,
            "empty_program_name": 0,
            "no_countersigner": 0,
            "parse_error": 0,
            "chain_max_depth": 0,
            "latest_signing_time": 0,
            "signing_time_diff": 0,
        }
        if SignedPEFile is None or signify is None:
            raw_obj["parse_error"] = 1
            return raw_obj
        try:
            signed_pe = SignedPEFile(io.BytesIO(bytez))
            for signed_data in signed_pe.iter_signed_datas():
                raw_obj["num_certs"] += 1
                if signed_data.signer_info.program_name is None:
                    raw_obj["empty_program_name"] = 1

                
                signer_info = signed_data.signer_info
                countersigner = signer_info.countersigner

                
                if countersigner is not None:
                    signing_time = countersigner.signing_time.timestamp()
                    if signing_time >= raw_obj["latest_signing_time"]:
                        raw_obj["latest_signing_time"] = signing_time
                    pe_timestamp = pe.FILE_HEADER.TimeDateStamp
                    raw_obj["signing_time_diff"] = signing_time - pe_timestamp
                else:
                    raw_obj["no_countersigner"] = 1

                
                certs = signed_data.certificates
                if len(certs) > raw_obj["chain_max_depth"]:
                    raw_obj["chain_max_depth"] = len(certs)
                for cert in certs[:-1]:
                    if cert.issuer == cert.subject:
                        raw_obj["self_signed"] = 1

        except signify.exceptions.SignerInfoParseError:
            raw_obj["parse_error"] = 1
        except signify.exceptions.ParseError:
            raw_obj["parse_error"] = 1
        except ValueError:
            raw_obj["parse_error"] = 1
        except KeyError:
            raw_obj["parse_error"] = 1
        return raw_obj

    def process_raw_features(self, raw_obj):
        if not raw_obj:
            return np.zeros(self.dim, dtype=np.float32)

        return np.hstack([
            raw_obj["num_certs"],
            raw_obj["self_signed"],
            raw_obj["empty_program_name"],
            raw_obj["no_countersigner"],
            raw_obj["parse_error"],
            raw_obj["chain_max_depth"],
            raw_obj["latest_signing_time"],
            raw_obj["signing_time_diff"],
        ]).astype(np.float32)


class PEFormatWarnings(FeatureType):
    """
    Features based on warnings thrown by PEFile parsing
    """

    name = "pefilewarnings"
    dim = 87 + 1

    def __init__(self, warnings_file: Path):
        self.warning_prefixes = set()
        self.warning_suffixes = set()
        self.warning_ids = {}
        
        if isinstance(warnings_file, Path) and warnings_file.exists():
            with open(warnings_file, "r") as f:
                i = 0
                for line in f:
                    line = line.strip()
                    if line.startswith("..."):
                        self.warning_suffixes.add(line[3:])
                        self.warning_ids[line] = i
                    else:
                        self.warning_prefixes.add(line[:-3])
                        self.warning_ids[line] = i
                    i += 1

    def raw_features(self, bytez, pe):
        if pe is None:
            return []

        warnings = set(pe.get_warnings())
        warnings_norm = set()
        for warning in warnings:
            found_warning = False
            for suf in self.warning_suffixes:
                if warning.endswith(suf):
                    warnings_norm.add("..." + suf)
                    found_warning = True
                    break
            if found_warning:
                continue
            for pre in self.warning_prefixes:
                if warning.startswith(pre):
                    warnings_norm.add(pre + "...")
                    found_warning = True
                    break
            if not found_warning:
                print("WARN: Unknown pefile warning:", warning)

        return sorted(warnings_norm)


    def process_raw_features(self, raw_obj):
        if not raw_obj:
            return np.zeros(self.dim, dtype=np.float32)

        ids = [0 for _ in range(self.dim)]
        for warning_norm in raw_obj:
            ids[self.warning_ids[warning_norm]] = 1.0
        ids[self.dim-1] = len(raw_obj)
        return np.array(ids, dtype=np.float32)


class PEFeatureExtractor(object):
    """
    NOTE:
    Day la "may trich xuat feature" chinh. No khong du doan malware.
    No chi doc byte cua file PE va bien cac thong tin nhu header, section,
    import/export, string, histogram... thanh mot vector so.
    """

    def __init__(self, features_file: Path | None = None):

        cwd = os.path.dirname(os.path.abspath(__file__))
        warnings_file = Path(os.path.join(cwd, "pefile_warnings.txt"))

        self.features = []
        # Danh sach cac nhom feature EMBER2024 se trich xuat.
        # Moi class ben duoi tao mot phan cua vector cuoi cung:
        #   GeneralFileInfo      : thong tin tong quat cua file.
        #   ByteHistogram        : tan suat byte 0-255.
        #   ByteEntropyHistogram : entropy theo tung vung byte.
        #   StringExtractor      : chuoi ASCII, URL, IP, command...
        #   HeaderFileInfo       : thong tin PE header.
        #   SectionInfo          : thong tin section .text/.data/.rsrc...
        #   ImportsInfo/ExportsInfo: import/export API.
        #   DataDirectories      : data directory cua PE.
        #   RichHeader, warnings : thong tin phu de nhan dien file bat thuong.
        features = OrderedDict([
            ("GeneralFileInfo", GeneralFileInfo()),
            ("ByteHistogram", ByteHistogram()),
            ("ByteEntropyHistogram", ByteEntropyHistogram()),
            ("StringExtractor", StringExtractor()),
            ("HeaderFileInfo", HeaderFileInfo()),
            ("SectionInfo", SectionInfo()),
            ("ImportsInfo", ImportsInfo()),
            ("ExportsInfo", ExportsInfo()),
            ("DataDirectories", DataDirectories()),
            ("RichHeader", RichHeader()),
            ("AuthenticodeSignature", AuthenticodeSignature()),
            ("PEFormatWarnings", PEFormatWarnings(warnings_file)),
        ])
        feature_names = features.keys()

        if isinstance(features_file, Path) and features_file.exists():
            with features_file.open(encoding="utf8") as f:
                x = json.load(f)
                self.features = [features[feature] for feature in feature_names if x["features"].get(feature) is not None]
        else:
            self.features = [features[feature] for feature in feature_names]

        self.dim = sum([fe.dim for fe in self.features])

    def raw_features(self, bytez: bytes):
        pe = None
        try:
            pe = pefile.PE(data=bytez)
        except pefile.PEFormatError:
            pass
        except AttributeError:
            pass
        features = {"sha256": hashlib.sha256(bytez).hexdigest()}
        features.update({fe.name: fe.raw_features(bytez, pe) for fe in self.features})
        return features

    def process_raw_features(self, raw_obj):
        feature_vectors = [fe.process_raw_features(raw_obj[fe.name]) for fe in self.features]
        return np.hstack(feature_vectors).astype(np.float32)

    def feature_vector(self, bytez):
        return self.process_raw_features(self.raw_features(bytez))
THREMBER_IMPORT_ERROR = None
_THREMBER_EXTRACTOR = None


def get_thrember_extractor():
    """Return the EMBER2024 feature schema expected by the trained 2560-feature model."""
    global _THREMBER_EXTRACTOR

    if _THREMBER_EXTRACTOR is None:
        # Tao extractor goc tu code EMBER2024 da nhung o phia tren.
        extractor = PEFeatureExtractor()

        # Model malware trong combined_ml_server.py chi nhan 2560 features.
        # Extractor goc tao 2568 features. Nhom "authenticode" co 8 features,
        # nen bo nhom nay de vector con dung 2560.
        extractor.features = [feature for feature in extractor.features if feature.name != "authenticode"]
        extractor.dim = sum(feature.dim for feature in extractor.features)

        # Kiem tra som de tranh gui sai so chieu sang model LightGBM.
        if extractor.dim != MODEL_META["num_features"]:
            raise RuntimeError(
                f"Feature dimension mismatch: thrember={extractor.dim}, "
                f"model={MODEL_META['num_features']}"
            )
        _THREMBER_EXTRACTOR = extractor

    return _THREMBER_EXTRACTOR




def extract_infos(fpath: str) -> list:
    """
    NOI TRICH XUAT FEATURES CHINH.

    Dau vao:
        fpath: duong dan file PE can phan tich.

    Dau ra:
        list[float] gom 2560 features, dung format combined_ml_server.py can.

    Luong xu ly:
        1. Mo file va doc toan bo bytes.
        2. Goi PEFeatureExtractor de trich xuat feature.
        3. Doi numpy array thanh list float de JSON serialize duoc khi POST.
    """
    with open(fpath, "rb") as f:
        raw = f.read()

    # Day la dong that su tao feature vector tu bytes cua file.
    vector = get_thrember_extractor().feature_vector(raw)

    # requests.post ben duoi can gui JSON, nen numpy array phai doi thanh list.
    return vector.astype(float).tolist()



def resolve_scan_dir() -> str:
    """
    Xac dinh thu muc se auto-scan.

    Neu co bien moi truong PE_SCAN_DIR thi uu tien bien do.
    Neu khong thi tim trong DEFAULT_SCAN_DIR_CANDIDATES.
    """
    env_dir = os.getenv("PE_SCAN_DIR")
    if env_dir:
        return env_dir

    for candidate in DEFAULT_SCAN_DIR_CANDIDATES:
        if os.path.isdir(candidate):
            return candidate

    return DEFAULT_SCAN_DIR_CANDIDATES[-1]


def is_pe_candidate(file_path: str) -> bool:
    """
    Kiem tra nhanh file co phai PE khong.

    File PE Windows thuong bat dau bang 2 byte "MZ".
    Ham nay giup bo qua file khong lien quan trong Downloads.
    """
    try:
        with open(file_path, "rb") as f:
            return f.read(2) == b"MZ"
    except Exception:
        return False


def get_local_agent_ip() -> str:
    """Return the primary IPv4 address used for outbound traffic."""
    env_ip = os.getenv("PE_AGENT_IP", "").strip()
    if env_ip:
        return env_ip

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            if ip and not ip.startswith("127."):
                return ip
    except Exception:
        pass

    try:
        for item in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = item[4][0]
            if ip and not ip.startswith("127."):
                return ip
    except Exception:
        pass

    return ""


def analyze_file(file_path: str, agent_name: str = "", agent_ip: str = "") -> dict:
    """
    HAM PHAN TICH 1 FILE CHO AUTO-SCAN / MONITOR.

    Luong xu ly:
        1. Lay ten may va IP agent.
        2. Goi extract_infos(file_path) de tao vector 2560 features.
        3. Dong goi payload JSON.
        4. Gui POST sang combined_ml_server.py endpoint /predict-features.
        5. Nhan ket qua label/probability tu combined.
    """
    agent_name = agent_name or socket.gethostname()
    agent_ip = agent_ip or get_local_agent_ip()

    # Buoc quan trong nhat: trich xuat feature tu file PE.
    feature_vector = extract_infos(file_path)
    if not feature_vector:
        raise RuntimeError("Empty feature vector")

    cat_check = {
        "is_pe":          feature_vector[2],
        "start_byte_0":   feature_vector[3],
        "start_byte_1":   feature_vector[4],
        "machine_type":   feature_vector[701],
        "subsystem":      feature_vector[702],
    }

    payload = {
        "features":    feature_vector,
        "path":        file_path,
        "agent_name":  agent_name,
        "agent_ip":    agent_ip,
    }

    # NOI GUI POST SANG COMBINED ML SERVER.
    # combined_ml_server.py se nhan payload nay tai endpoint /predict-features.
    resp = requests.post(ML_SERVER_URL, json=payload, timeout=10)
    resp.raise_for_status()

    ml_result = resp.json()
    ml_result["categorical_check"] = cat_check
    ml_result["feature_dim"] = len(feature_vector)
    return ml_result


def scan_directory(scan_dir: str) -> dict:
    """
    QUET THU MUC KHI SERVICE VUA START.

    Ham nay di qua tung file trong scan_dir:
        - Neu file khong bat dau bang "MZ" thi bo qua.
        - Neu la PE candidate thi goi analyze_file().
        - Ket qua se duoc in ra console.
    """
    summary = {
        "scan_dir": scan_dir,
        "scanned": 0,
        "skipped": 0,
        "errors": 0,
        "results": [],
    }

    if not os.path.isdir(scan_dir):
        print(f"[!] Scan directory not found: {scan_dir}")
        return summary

    agent_name = os.getenv("PE_AGENT_NAME", socket.gethostname())
    agent_ip = get_local_agent_ip()

    print(f"[+] Auto scan enabled: {scan_dir}")
    for root, _, files in os.walk(scan_dir):
        for name in files:
            file_path = os.path.join(root, name)
            if not is_pe_candidate(file_path):
                summary["skipped"] += 1
                continue

            try:
                result = analyze_file(file_path, agent_name=agent_name, agent_ip=agent_ip)
                summary["scanned"] += 1
                summary["results"].append({
                    "path": file_path,
                    "label": result.get("label"),
                    "probability": result.get("probability"),
                })
                print(
                    f"[+] Scanned: {file_path} | "
                    f"label={result.get('label')} prob={result.get('probability')}"
                )
            except Exception as e:
                summary["errors"] += 1
                print(f"[!] Failed: {file_path} | {e}")

    print(
        f"[+] Auto scan finished | scanned={summary['scanned']} "
        f"skipped={summary['skipped']} errors={summary['errors']}"
    )
    return summary


def file_scan_key(file_path: str) -> tuple:
    """Use stable file metadata so changed files can be scanned again."""
    st = os.stat(file_path)
    return (os.path.abspath(file_path).lower(), st.st_size, int(st.st_mtime))


def monitor_directory(scan_dir: str, interval_seconds: int = 10) -> None:
    """
    THEO DOI THU MUC LIEN TUC.

    Khac voi scan_directory() chi chay mot lan, ham nay chay vong lap vo han.
    Moi interval_seconds giay, no se tim file moi/file thay doi va phan tich.
    """
    seen = set()
    agent_name = os.getenv("PE_AGENT_NAME", socket.gethostname())
    agent_ip = get_local_agent_ip()

    print(f"[+] Directory monitor enabled: {scan_dir} every {interval_seconds}s")
    while True:
        if not os.path.isdir(scan_dir):
            time.sleep(interval_seconds)
            continue

        for root, _, files in os.walk(scan_dir):
            for name in files:
                file_path = os.path.join(root, name)
                try:
                    key = file_scan_key(file_path)
                except Exception:
                    continue

                if key in seen:
                    continue
                seen.add(key)

                if not is_pe_candidate(file_path):
                    continue

                try:
                    result = analyze_file(file_path, agent_name=agent_name, agent_ip=agent_ip)
                    print(
                        f"[+] Monitor scanned: {file_path} | "
                        f"label={result.get('label')} prob={result.get('probability')}"
                    )
                except Exception as e:
                    print(f"[!] Monitor failed: {file_path} | {e}")

        time.sleep(interval_seconds)


def start_directory_monitor(scan_dir: str) -> None:
    """Start the background directory monitor without blocking the Flask service."""
    thread = threading.Thread(
        target=monitor_directory,
        args=(scan_dir, WATCH_INTERVAL_SECONDS),
        daemon=True,
    )
    thread.start()




@app.route("/analyze", methods=["POST"])
def analyze():
    """
    API FLASK CHO BEN NGOAI GOI VAO.

    Cach goi:
        POST http://<pe-service-host>:5001/analyze

    Body JSON mau:
        {
            "file_path": "C:\\path\\file.exe",
            "agent_name": "DESKTOP-ABC",
            "agent_ip": "192.168.x.x"
        }

    Ham nay lam giong analyze_file(), nhung duoc kich hoat bang HTTP request
    thay vi auto-scan.
    """
    data = request.get_json()

    file_path  = data.get("file_path")
    agent_name = data.get("agent_name", "")
    agent_ip   = data.get("agent_ip", "") or get_local_agent_ip()

    if not file_path or not os.path.isfile(file_path):
        return jsonify({"error": "File path invalid or file not found"}), 400

    # Trich xuat 2560 features tu file_path client gui len.
    try:
        feature_vector = extract_infos(file_path)
    except Exception as e:
        return jsonify({"error": f"Feature extraction failed: {str(e)}"}), 500

    if not feature_vector:
        return jsonify({"error": "Empty feature vector"}), 500

    # Sanity check: log categorical values
    cat_check = {
        "is_pe":          feature_vector[2],
        "start_byte_0":   feature_vector[3],
        "start_byte_1":   feature_vector[4],
        "machine_type":   feature_vector[701],
        "subsystem":      feature_vector[702],
    }

    # Payload nay se duoc gui sang combined_ml_server.py /predict-features.
    payload = {
        "features":    feature_vector,
        "path":        file_path,
        "agent_name":  agent_name,
        "agent_ip":    agent_ip,
    }
    try:
        # NOI GUI POST SANG COMBINED ML SERVER KHI GOI API /analyze.
        resp = requests.post(ML_SERVER_URL, json=payload, timeout=10)
        resp.raise_for_status()
        ml_result = resp.json()
        ml_result["categorical_check"] = cat_check
        ml_result["feature_dim"]       = len(feature_vector)
        return jsonify(ml_result)
    except Exception as e:
        return jsonify({
            "error":           str(e),
            "feature_dim":     len(feature_vector),
            "categorical_check": cat_check,
        }), 500


if __name__ == "__main__":
    # Khi chay truc tiep:
    #   python "pe_extractor_service (1).py"
    #
    # Thu tu chay:
    #   1. Xac dinh thu muc scan.
    #   2. Neu AUTO_SCAN_ON_START=1 thi scan Downloads mot lan.
    #   3. Neu WATCH_SCAN_DIR=1 thi bat thread theo doi Downloads.
    #   4. Mo Flask service tai port PE_SERVICE_PORT, mac dinh 5001.
    scan_dir = resolve_scan_dir()

    if AUTO_SCAN_ON_START:
        scan_directory(scan_dir)

    if WATCH_SCAN_DIR:
        start_directory_monitor(scan_dir)

    service_host = os.getenv("PE_SERVICE_HOST", "0.0.0.0")
    service_port = int(os.getenv("PE_SERVICE_PORT", "5001"))
    app.run(host=service_host, port=service_port, debug=False)
