"""
═══════════════════════════════════════════════════════════════
通用多分类漏洞数据集整合模板 (最终版)
标签体系（与 TrVD / MTVD 完全一致）：
  0       → 正常样本（无漏洞）
  1~85    → 对应 85 种漏洞类型（CWE编号见映射表）

过滤条件：
  1. 代码长度在 [MIN_FUNC_LEN, MAX_FUNC_LEN] 范围内
  2. 最大花括号嵌套深度 >= MIN_NEST_DEPTH（去除注释和字符串后计算）

使用方法：
  1. 修改 ═══ 用户配置��� ═══ 中的路径和参数
  2. 为每个数据集实现对应的 Loader 类（模板已提供）
  3. 运行脚本
═══════════════════════════════════════════════════════════════
"""

import os
import re
import json
import pickle
import hashlib
import pandas as pd
from abc import ABC, abstractmethod
from collections import Counter
from tqdm import tqdm


# ================================================================
# ════════════════════════ 用户配置区 ════════════════════════════
# ================================================================

OUTPUT_PATH    = "E:/TH/thpaper1/multiclass_dataset.jsonl"

MIN_FUNC_LEN   = 50       # 函数最小字符数
MAX_FUNC_LEN   = 500000000    # 函数最大字符数
MIN_NEST_DEPTH = 2        # 最小嵌套深度（2 = 函数体内至少有一层 if/for/while）

#数据集集合
DATASET_CONFIGS = [
    {
        "name":   "BigVul",
        "path":   "E:/TH/thpaper1/DataBase/MSR_data_cleaned/MSR_data_cleaned.csv",
        "fmt":    "csv",
        "loader": "BigVulLoader",
    },
    {
        "name":   "TrVD",
        "path":   "E:/TH/thpaper1/DataBase/TrVD-main/dataset/dataset/dataset.pkl",
        "fmt":    "pkl",
        "loader": "TrVDLoader",
    },
    {
        "name":   "DiverseVul",
        "path":   "E:/TH/thpaper1/diversevul_20230702.json",
        "fmt":    "jsonl",
        "loader": "DiverseVulLoader",
    },
    # 新增数据集照此格式添加：
    # {
    #     "name":   "MyDataset",
    #     "path":   "data/my_dataset.csv",
    #     "fmt":    "csv",
    #     "loader": "MyDatasetLoader",
    # },
]

# ================================================================
# ══════════════════ 映射表（勿修改）════════════════════════════
# ================================================================

LABEL_TO_CWE: dict[int, str] = {
    1:  "CWE-22",           2:  "CWE-73",           3:  "CWE-77",
    4:  "CWE-119",          5:  "CWE-134",           6:  "CWE-176",
    7:  "CWE-190",          8:  "CWE-191",           9:  "CWE-200",
    10: "CWE-252",          11: "CWE-253",           12: "CWE-271",
    13: "CWE-284",          14: "CWE-319",           15: "CWE-325",
    16: "CWE-327",          17: "CWE-328",           18: "CWE-362",
    19: "CWE-369",          20: "CWE-377",           21: "CWE-390",
    22: "CWE-391",          23: "CWE-396",           24: "CWE-398",
    25: "CWE-400",          26: "CWE-404",           27: "CWE-426",
    28: "CWE-427",          29: "CWE-459",           30: "CWE-464",
    31: "CWE-467",          32: "CWE-468",           33: "CWE-469",
    34: "CWE-476",          35: "CWE-480",           36: "CWE-534",
    37: "CWE-563",          38: "CWE-588",           39: "CWE-606",
    40: "CWE-617",          41: "CWE-628",           42: "CWE-665",
    43: "CWE-666",          44: "CWE-667",           45: "CWE-672",
    46: "CWE-675",          47: "CWE-681",           48: "CWE-690",
    49: "CWE-758",          50: "CWE-763",           51: "CWE-771",
    52: "CWE-772",          53: "CWE-789",           54: "CWE-843",
    55: "CWE-908",          56: "CWE-912",           57: "CWE-943",
    58: "CWE-1078",         59: "CWE-1105",          60: "CWE-1177",
    61: "CWE-119,672",      62: "CWE-119,672,415",
    63: "CWE-1390,344",     64: "CWE-1390,522",
    65: "CWE-15,642",       66: "CWE-459,212",
    67: "CWE-222",          68: "CWE-223",           69: "CWE-247",
    70: "CWE-273",          71: "CWE-338",           72: "CWE-397",
    73: "CWE-475",          74: "CWE-478",           75: "CWE-479",
    76: "CWE-483",          77: "CWE-484",           78: "CWE-535",
    79: "CWE-570",          80: "CWE-571",           81: "CWE-587",
    82: "CWE-605",          83: "CWE-620",           84: "CWE-785",
    85: "CWE-835",
}

CWE_TO_LABEL: dict[str, int] = {v: k for k, v in LABEL_TO_CWE.items()}

SINGLE_CWE_TO_LABEL: dict[str, int] = {}
for _lbl, _cwe_str in LABEL_TO_CWE.items():
    for _part in _cwe_str.split(","):
        _key = _part.strip()
        if not _key.startswith("CWE-"):
            _key = f"CWE-{_key}"
        if _key not in SINGLE_CWE_TO_LABEL:
            SINGLE_CWE_TO_LABEL[_key] = _lbl


# ================================================================
# ════════════════ 核心工具函数（勿修改）═════════════════════════
# ================================================================

# ── 1. 注释/字符串剥离 ──────────────────────────────────────────
def _strip_comments_and_strings(code: str) -> str:
    """去除 C/C++ 代码中的注释和字符串字面量，避免干扰深度计数"""
    result = []
    i, n = 0, len(code)
    while i < n:
        if code[i:i+2] == '/*':                    # 块注释
            end = code.find('*/', i + 2)
            i = end + 2 if end != -1 else n
        elif code[i:i+2] == '//':                  # 行注释
            end = code.find('\n', i + 2)
            i = end + 1 if end != -1 else n
        elif code[i] == '"':                        # 双引号字符串
            i += 1
            while i < n:
                if code[i] == '\\': i += 2
                elif code[i] == '"': i += 1; break
                else: i += 1
        elif code[i] == "'":                        # 字符字面量
            i += 1
            while i < n:
                if code[i] == '\\': i += 2
                elif code[i] == "'": i += 1; break
                else: i += 1
        else:
            result.append(code[i])
            i += 1
    return ''.join(result)


# ── 2. 嵌套深度计算 ────────────────────────────────────────────
def get_nesting_depth(code: str) -> int:
    """
    计算 C/C++ 函数的最大花括号嵌套深度（已过滤注释和字符串）
    depth=1 → 只有函数体，无嵌套
    depth=2 → 函数体内有一层 if/for/while 嵌套
    depth=N → N-1 层嵌套控制流
    """
    cleaned   = _strip_comments_and_strings(code)
    max_depth = 0
    depth     = 0
    for ch in cleaned:
        if ch == '{':
            depth += 1
            if depth > max_depth:
                max_depth = depth
        elif ch == '}':
            depth -= 1
            if depth < 0:
                depth = 0
    return max_depth


# ── 3. 综合过滤 ────────────────────────────────────────────────
def is_valid_func(code: str,
                  min_len:   int = MIN_FUNC_LEN,
                  max_len:   int = MAX_FUNC_LEN,
                  min_depth: int = MIN_NEST_DEPTH) -> bool:
    """
    综合过滤函数，同时检查：
      ① 类型非空
      ② 长度在 [min_len, max_len] 内
      ③ 嵌套深度 >= min_depth（默认 2，即至少有一层嵌套）
    """
    if not isinstance(code, str):
        return False
    c = code.strip()
    if not (min_len <= len(c) <= max_len):
        return False
    return get_nesting_depth(c) >= min_depth


# ── 4. CWE 解析 ────────────────────────────────────────────────
def parse_cwe_to_label(cwe_raw) -> int:
    """
    将各种格式的 CWE 值解析为 label（1-85）
    支持：
      "CWE-119" / "119" / 119 / ["CWE-119","CWE-672"] / "CWE-119,CWE-672"
    返回 -1 表示不在映射表内
    """
    if cwe_raw is None or cwe_raw == "":
        return -1
    if isinstance(cwe_raw, list):
        parts = [str(c).strip() for c in cwe_raw if str(c).strip()]
    elif isinstance(cwe_raw, (int, float)):
        parts = [f"CWE-{int(cwe_raw)}"]
    elif isinstance(cwe_raw, str):
        parts = [c.strip() for c in cwe_raw.split(",") if c.strip()]
    else:
        return -1

    normalized = []
    for p in parts:
        if re.match(r'^\d+$', p):
            normalized.append(f"CWE-{p}")
        elif re.match(r'^cwe-\d+', p, re.IGNORECASE):
            num = re.search(r'(\d+)', p).group(1)
            normalized.append(f"CWE-{num}")
        else:
            normalized.append(p)

    # 优先复合 CWE 精确匹配
    nums = [re.search(r'(\d+)', n).group(1)
            for n in normalized if re.search(r'(\d+)', n)]
    if len(nums) >= 2:
        for composite in [",".join(nums),
                          ",".join(f"CWE-{n}" for n in nums)]:
            if composite in CWE_TO_LABEL:
                return CWE_TO_LABEL[composite]

    # 单 CWE 逐个匹配
    for n in normalized:
        if n in SINGLE_CWE_TO_LABEL:
            return SINGLE_CWE_TO_LABEL[n]

    return -1


# ── 5. 其他工具 ────────────────────────────────────────────────
def normalize_code(code: str) -> str:
    if not isinstance(code, str):
        return ""
    code = re.sub(r'/\*.*?\*/', ' ', code, flags=re.DOTALL)
    code = re.sub(r'//.*', ' ', code)
    return re.sub(r'\s+', ' ', code).strip()


def code_hash(code: str) -> str:
    return hashlib.md5(
        normalize_code(code).encode("utf-8", errors="ignore")
    ).hexdigest()


def make_record(idx, source, func,
                label, cve_id="", cwe_raw="", project="") -> dict:
    """统一记录构造，所有 Loader 均调用此函数"""
    return {
        "id":      f"{source}_{idx}",
        "source":  source,
        "func":    func.strip(),
        "label":   label,
        "cve_id":  cve_id,
        "cwe_raw": cwe_raw,
        "project": project,
    }


# ================================================================
# ══════════════════ Loader 基类（勿修改）════════════════════════
# ================================================================

class BaseLoader(ABC):
    """
    数据集 Loader 基类
    子类实现 load() → yield/append make_record(...) 即可

    label 规则：
      正常样本               → label = 0
      漏洞且 CWE 已知        → label = parse_cwe_to_label(cwe)   (1-85)
      漏洞但 CWE 未知/不在表 → label = -1
      TrVD 直接使用原始label  → label = 原始值 (0-85)
    """
    def __init__(self, config: dict):
        self.name = config["name"]
        self.path = config["path"]
        self.fmt  = config["fmt"]

    @abstractmethod
    def load(self) -> list[dict]:
        ...


# ================================================================
# ══════════════════ Loader 实现区（用户按需修改）═════════════════
# ================================================================

class BigVulLoader(BaseLoader):
    """
    BigVul: MSR_data_cleaned.csv
    关键列: func_before / func_after / vul(0/1) / CWE ID / CVE ID / project
    """
    def load(self) -> list[dict]:
        records = []
        reader = pd.read_csv(
            self.path,
            chunksize=50_000,
            quoting=0,
            on_bad_lines="skip",
            low_memory=False,
            usecols=["Unnamed: 0", "func_before", "func_after",
                     "vul", "CVE ID", "CWE ID", "project"],
        )
        for chunk in tqdm(reader, desc=f"  {self.name} chunks"):
            chunk = chunk.rename(columns={
                "Unnamed: 0": "id",
                "CVE ID":     "cve_id",
                "CWE ID":     "cwe_raw",
            })
            for _, row in chunk.iterrows():
                vul     = int(row.get("vul", 0))
                cwe_raw = row.get("cwe_raw", "")
                cve_id  = str(row.get("cve_id","")) if pd.notna(row.get("cve_id")) else ""
                project = str(row.get("project","")) if pd.notna(row.get("project")) else ""
                rid     = str(row.get("id", ""))

                # func_before
                func_b = row.get("func_before", "")
                if is_valid_func(func_b):
                    if vul == 1:
                        label = (parse_cwe_to_label(cwe_raw)
                                 if pd.notna(cwe_raw) and cwe_raw != ""
                                 else -1)
                    else:
                        label = 0
                    records.append(make_record(
                        idx=rid, source=self.name.lower(),
                        func=str(func_b), label=label,
                        cve_id=cve_id,
                        cwe_raw=str(cwe_raw) if pd.notna(cwe_raw) else "",
                        project=project,
                    ))

                # func_after → 修复后代码，label=0
                func_a = row.get("func_after", "")
                if (is_valid_func(func_a)
                        and str(func_a).strip() != str(func_b).strip()):
                    records.append(make_record(
                        idx=f"{rid}_fixed", source=self.name.lower(),
                        func=str(func_a), label=0,
                        cve_id=cve_id, project=project,
                    ))
        return records


class TrVDLoader(BaseLoader):
    """
    TrVD: dataset.pkl
    关键列: code / label（0=正常，1-85=漏洞类型，与映射表完全一致）
    """
    def load(self) -> list[dict]:
        with open(self.path, "rb") as f:
            raw = pickle.load(f)
        df = raw if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw)

        records = []
        for idx, row in tqdm(df.iterrows(), total=len(df),
                             desc=f"  {self.name} rows"):
            code  = row.get("code", "")
            label = int(row.get("label", 0))
            if not is_valid_func(code) or not (0 <= label <= 85):
                continue
            records.append(make_record(
                idx=str(idx), source=self.name.lower(),
                func=str(code), label=label,
                cwe_raw=LABEL_TO_CWE.get(label, "") if label > 0 else "",
            ))
        return records


class DiverseVulLoader(BaseLoader):
    """
    DiverseVul: JSONL 格式
    关键字段: func / target(0/1) / cwe(list) / project
    """
    def load(self) -> list[dict]:
        records = []
        with open(self.path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(tqdm(f, desc=f"  {self.name} lines")):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue

                func = item.get("func", "")
                if not is_valid_func(func):
                    continue

                target  = int(item.get("target", 0))
                cwe_raw = item.get("cwe", [])

                if target == 0:
                    label, cwe_str = 0, ""
                else:
                    label   = parse_cwe_to_label(cwe_raw)
                    cwe_str = (",".join(cwe_raw) if isinstance(cwe_raw, list)
                               else str(cwe_raw))

                records.append(make_record(
                    idx=str(idx), source=self.name.lower(),
                    func=str(func), label=label,
                    cwe_raw=cwe_str,
                    project=str(item.get("project", "")),
                ))
        return records


# ================================================================
# 📝 新增数据集模板（复制此类，修改 load() 即可）
# ================================================================

class MyDatasetLoader(BaseLoader):
    """
    自定义数据集模板
    ─────────────────────────────────────────────
    可用工具函数：
      is_valid_func(code)          → 长度 + 深度综合过滤
      get_nesting_depth(code)      → 只获取深度数值
      parse_cwe_to_label(cwe)      → CWE字符串/list → label(1-85)
      make_record(idx, source, ...) → 构造标准记录
      LABEL_TO_CWE[label]          → label → CWE字符串
    ─────────────────────────────────────────────
    """
    def load(self) -> list[dict]:
        records = []

        # ── 情形 A：CSV，label 已经是 0-85 ─────────────────────
        # df = pd.read_csv(self.path)
        # for idx, row in df.iterrows():
        #     code  = str(row["code_column"])
        #     label = int(row["label_column"])     # 已是 0-85，直接用
        #     if not is_valid_func(code):
        #         continue
        #     records.append(make_record(
        #         idx=str(idx), source=self.name.lower(),
        #         func=code, label=label,
        #         cwe_raw=LABEL_TO_CWE.get(label, ""),
        #     ))

        # ── 情形 B：CSV，需从 CWE 字符串解析 label ──────────────
        # df = pd.read_csv(self.path)
        # for idx, row in df.iterrows():
        #     code = str(row["func"])
        #     cwe  = str(row.get("cwe", ""))
        #     is_vul = int(row.get("is_vulnerable", 0))
        #     if not is_valid_func(code):
        #         continue
        #     label = parse_cwe_to_label(cwe) if is_vul else 0
        #     records.append(make_record(
        #         idx=str(idx), source=self.name.lower(),
        #         func=code, label=label, cwe_raw=cwe,
        #     ))

        # ── 情形 C：JSONL ────────────────────────────────────────
        # with open(self.path, "r", encoding="utf-8") as f:
        #     for idx, line in enumerate(f):
        #         item = json.loads(line.strip())
        #         code = item.get("func", "")
        #         if not is_valid_func(code):
        #             continue
        #         records.append(make_record(
        #             idx=str(idx), source=self.name.lower(),
        #             func=code,
        #             label=parse_cwe_to_label(item.get("cwe", "")),
        #             cwe_raw=str(item.get("cwe", "")),
        #         ))

        return records


# ================================================================
# ════════════════ 去重 & 统计 & 主流程（勿修改）══════════════════
# ================================================================

def merge_and_dedup(all_records: list[dict]) -> list[dict]:
    print("\n🔄 去重中 ...")

    def priority(r: dict) -> int:
        s = 0
        if r.get("cve_id"):               s += 2
        if r.get("cwe_raw"):              s += 1
        if 1 <= r.get("label", -1) <= 85: s += 2
        if r.get("label", -1) == 0:       s += 1
        return s

    seen: dict[str, dict] = {}
    for rec in tqdm(all_records, desc="  去重"):
        h = code_hash(rec["func"])
        if h not in seen or priority(rec) > priority(seen[h]):
            seen[h] = rec
    return list(seen.values())


def print_stats(records: list[dict]):
    label_counter = Counter(r["label"] for r in records)
    total     = len(records)
    normal    = label_counter.get(0, 0)
    unknown   = label_counter.get(-1, 0)
    known_vul = sum(v for k, v in label_counter.items() if 1 <= k <= 85)
    covered   = sum(1 for lbl in range(1, 86) if label_counter.get(lbl, 0) > 0)

    print(f"\n{'═'*62}")
    print(f"  📊 多分类数据集统计（过滤：深度≥{MIN_NEST_DEPTH} / 长度{MIN_FUNC_LEN}~{MAX_FUNC_LEN}）")
    print(f"{'═'*62}")
    print(f"  总计                 : {total:>10,} 条")
    print(f"  正常样本  (label= 0) : {normal:>10,} 条  ({normal/total:.2%})")
    print(f"  已知漏洞  (label1-85): {known_vul:>10,} 条  ({known_vul/total:.2%})")
    print(f"  未知CWE   (label=-1) : {unknown:>10,} 条  ({unknown/total:.2%})")
    print(f"  覆盖漏洞类型数       : {covered}/85")

    print(f"\n  {'─'*58}")
    print(f"  {'标签':>5}  {'CWE':^22}  {'数量':>8}  {'占漏洞%':>8}")
    print(f"  {'─'*58}")
    for lbl in range(1, 86):
        cnt = label_counter.get(lbl, 0)
        if cnt == 0:
            continue
        cwe = LABEL_TO_CWE.get(lbl, "?")
        pct = cnt / known_vul if known_vul > 0 else 0
        bar = "▓" * min(int(pct * 200), 20)
        print(f"  {lbl:>5}  {cwe:<22}  {cnt:>8,}  {pct:>7.2%}  {bar}")

    print(f"\n  {'─'*58}")
    print(f"  📦 来源分布：")
    for src, cnt in Counter(r["source"] for r in records).most_common():
        print(f"    {src:<15}: {cnt:>8,} 条  ({cnt/total:.2%})")


def save_jsonl(records: list[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  💾 已保存: {path}  ({len(records):,} 条)")


def get_loader(config: dict) -> BaseLoader:
    loader_map = {cls.__name__: cls for cls in BaseLoader.__subclasses__()}
    cls_name = config.get("loader", "")
    if cls_name not in loader_map:
        raise ValueError(
            f"未找到 Loader: '{cls_name}'，可用: {list(loader_map.keys())}"
        )
    return loader_map[cls_name](config)


def main():
    print("=" * 62)
    print("  🚀 通用多分类漏洞数据集整合（label 0-85）")
    print(f"  过滤参数: 长度[{MIN_FUNC_LEN},{MAX_FUNC_LEN}] / 嵌套深度≥{MIN_NEST_DEPTH}")
    print("=" * 62)

    all_records = []
    for cfg in DATASET_CONFIGS:
        print(f"\n[{DATASET_CONFIGS.index(cfg)+1}/{len(DATASET_CONFIGS)}]"
              f" 加载 {cfg['name']} ...")
        loader  = get_loader(cfg)
        records = loader.load()
        print(f"  {cfg['name']} 有效记录（深度过滤后）: {len(records):,}")
        all_records.extend(records)

    print(f"\n  合并前总量: {len(all_records):,} 条")
    dedup = merge_and_dedup(all_records)
    print(f"  去重后总量: {len(dedup):,} 条  "
          f"(去除 {len(all_records)-len(dedup):,} 条)")

    print_stats(dedup)

    known   = [r for r in dedup if r["label"] >= 0]
    unknown = [r for r in dedup if r["label"] == -1]

    print()
    save_jsonl(known, OUTPUT_PATH)
    if unknown:
        unk_path = OUTPUT_PATH.replace(".jsonl", "_unknown_cwe.jsonl")
        save_jsonl(unknown, unk_path)
        print(f"  ⚠️  未知CWE另存: {unk_path}  ({len(unknown):,} 条)")

    print("\n" + "=" * 62)
    print("  ✅ 完成！")
    print("=" * 62)


if __name__ == "__main__":
    main()