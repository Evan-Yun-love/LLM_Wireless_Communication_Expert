ACRONYM_GLOSSARY = {
    # --- Layer 1: Physical Layer ---
    "PHY": "Physical Layer",
    "CP": "Cyclic Prefix",
    "FFT": "Fast Fourier Transform",
    "IFFT": "Inverse Fast Fourier Transform",
    "OFDM": "Orthogonal Frequency Division Multiplexing",
    "FDMA": "Frequency Division Multiple Access",
    "DMRS": "Demodulation Reference Signal",
    "SRS": "Sounding Reference Signal",
    "PSS": "Primary Synchronization Signal",
    "SSS": "Secondary Synchronization Signal",
    "PBCH": "Physical Broadcast Channel",
    "PDSCH": "Physical Downlink Shared Channel",
    "PUSCH": "Physical Uplink Shared Channel",
    "PRACH": "Physical Random Access Channel",
    "PUCCH": "Physical Uplink Control Channel",
    "PDCCH": "Physical Downlink Control Channel",
    "CSI": "Channel State Information",
    "RS": "Reference Signal",
    "RMS": "Root Mean Square",
    "SC": "Single Carrier",
    "MC": "Multi Carrier",
    
    # --- Layer 2: MAC/RLC/PDCP ---
    "MAC": "Medium Access Control",
    "RLC": "Radio Link Control",
    "PDCP": "Packet Data Convergence Protocol",
    "HARQ": "Hybrid Automatic Repeat Request",
    "RNTI": "Radio Network Temporary Identifier",
    "LCID": "Logical Channel ID",
    "TBS": "Transport Block Size",
    "TTI": "Transmission Time Interval",
    "BWP": "Bandwidth Part",
    "RB": "Resource Block",
    "RE": "Resource Element",

    # --- Layer 3: RRC/NAS ---
    "RRC": "Radio Resource Control",
    "NAS": "Non-Access Stratum",
    "AS": "Access Stratum",
    "MME": "Mobility Management Entity",
    "AMF": "Access and Mobility Management Function",
    "SMF": "Session Management Function",
    "UE": "User Equipment",
    "BS": "Base Station",
    "gNB": "Next-generation Node B",
    "eNB": "Evolved Node B",
    "RAT": "Radio Access Technology",

    # --- Multiple Access & Duplex ---
    "OFDMA": "Orthogonal Frequency Division Multiple Access",
    "TDMA": "Time Division Multiple Access",
    "CDMA": "Code Division Multiple Access",
    "TDD": "Time Division Duplex",
    "FDD": "Frequency Division Duplex",

    # --- Antenna/Beamforming ---
    "MIMO": "Multiple Input Multiple Output",
    "SU-MIMO": "Single User MIMO",
    "MU-MIMO": "Multi-User MIMO",
    "BF": "Beamforming",

    # --- Control Channels/Data Channels ---
    "DCI": "Downlink Control Information",
    "BCCH": "Broadcast Control Channel",
    "PCCH": "Paging Control Channel",
    "CCCH": "Common Control Channel",
    "MCCH": "Multicast Control Channel",
    "MTCH": "Multicast Traffic Channel",
    "DL-SCH": "Downlink Shared Channel",
    "UL-SCH": "Uplink Shared Channel",

    # --- Measurements and Reports ---
    "CQI": "Channel Quality Indicator",
    "PMI": "Precoding Matrix Indicator",
    "RI": "Rank Indicator",
    "TA": "Timing Advance",

    # --- Access & Mobility ---
    "RA": "Random Access",
    "HO": "Handover",
    "PLMN": "Public Land Mobile Network",
    "TAI": "Tracking Area Identity",
    "ECGI": "E-UTRAN Cell Global Identifier",
    "IMSI": "International Mobile Subscriber Identity",
    "IMEI": "International Mobile Equipment Identity",
    "GUTI": "Globally Unique Temporary UE Identity",

    # --- Core Network ---
    "EPC": "Evolved Packet Core",
    "NGC": "Next Generation Core",
    "UPF": "User Plane Function",
    "SGW": "Serving Gateway",
    "PGW": "Packet Data Network Gateway",
    "PCRF": "Policy and Charging Rules Function",

    # --- Security ---
    "ASME": "Access Security Management Entity",
    "KSI": "Key Set Identifier",
    "NAS-SMC": "NAS Security Mode Command",

    # --- Misc ---
    "NR": "New Radio",
    "5GC": "5G Core",
    "QCI": "QoS Class Identifier",
    "SCTP": "Stream Control Transmission Protocol",
    "IP": "Internet Protocol",
    "QoS": "Quality of Service"
}




import re
from collections import defaultdict
from typing import List, Tuple, Dict, Union

from typing import List, Tuple, Dict, Union
from collections import defaultdict
import re

def expand_acronyms(text: str, glossary: dict = ACRONYM_GLOSSARY) -> str:
    for acronym, full_form in glossary.items():
        pattern = r"\b" + re.escape(acronym) + r"\b"
        replacement = f"{full_form} ({acronym})"
        text = re.sub(pattern, replacement, text)
    return text

def format_context_grouped(
    results: List[Tuple[str, float, dict]],
    with_metadata: bool = True,
    with_score: bool = False,
    separator: str = "\n\n---\n\n",
    glossary: dict = ACRONYM_GLOSSARY  # 新增参数
) -> str:
    """
    按文档+页码聚合格式化搜索结果，并在上下文中扩展术语缩写。

    参数:
        results: 检索返回的三元组列表 (chunk_text, score, metadata)
        with_metadata: 是否加文档名与页码信息作为前缀
        with_score: 是否在段首显示得分
        separator: 各段落之间的分隔符
        glossary: 可选术语字典 {缩写: 全称}

    返回:
        格式化后的 context 字符串
    """
    grouped: Dict[Tuple[str, Union[int, str]], List[Tuple[str, float]]] = defaultdict(list)

    for text, score, meta in results:
        doc = meta.get("document", "unknown_doc")
        page = meta.get("page", "unknown_page")
        grouped[(doc, page)].append((text.strip(), score))

    chunks = []
    for (doc, page), texts_scores in sorted(grouped.items(), key=lambda x: (x[0][0], str(x[0][1]))):
        prefix = f"[Document: {doc}, Page: {page}]" if with_metadata else ""
        merged_text = "\n".join([
            f"(score={score:.3f}) {text}" if with_score else text
            for text, score in texts_scores
        ])
        # 应用术语扩展（可选）
        # if glossary is not None:
        #     merged_text = expand_acronyms(merged_text, glossary)

        chunk = f"{prefix}\n{merged_text}" if prefix else merged_text
        chunks.append(chunk)

    return separator.join(chunks)
