from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ZERO_WIDTH_CHARS = ["\u200b", "\u200c", "\u200d", "\ufeff", "\u2060"]
NBSP_CHARS = ["\u00a0", "\u202f", "\u2007"]


# ----------------------------
# Header/footer removal
# ----------------------------

DEFAULT_HEADER_FOOTER_REGEXES = [
    # Page markers
    r"^\s*page\s+\d+(\s+of\s+\d+)?\s*$",
    r"^\s*\d+\s*/\s*\d+\s*$",

    # EU Official Journal-ish patterns (often show up in GDPR PDFs)
    r"^\s*official\s+journal\s+of\s+the\s+european\s+union\s*$",
    r"^\s*o\s*j\s+l\s+\d+.*$",
    r"^\s*l\s+\d+\s*/\s*\d+\s*$",
    r"^\s*(en|fr|de)\s*$",  # language code lines
    r"^\s*\d{1,2}\.\d{1,2}\.\d{4}\s*$",  # dd.mm.yyyy
    r"^\s*\(\s*text\s+with\s+eea\s+relevance\s*\)\s*$",

    # Generic “downloaded from” / watermark type lines
    r"^\s*downloaded\s+from\s+.*$",
    r"^\s*printed\s+on\s+.*$",
    r"^\s*copyright\s+.*$",
]

# Avoid removing real content that happens to repeat
PROTECT_LINE_REGEXES = [
    r"^\s*(chapter|section|article)\b",  # headings
    r"^\s*\d+\.\s*$",                    # clause numbers like "1."
    r"^\s*\(\s*\d+\s*\)\s*$",            # "(1)"
]


def _norm_line_for_count(line: str) -> str:
    """
    Normalize a line for frequency counting.
    - lower
    - collapse whitespace
    - mask digits (page numbers / dates / line refs)
    """
    s = line.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\d", "#", s)
    return s


def strip_headers_footers(
    text: str,
    *,
    min_repeats: int = 4,
    max_line_len: int = 140,
    regexes: Optional[List[str]] = None,
    protect_regexes: Optional[List[str]] = None,
) -> str:
    """
    Remove likely headers/footers:
      A) regex-based removal for common patterns
      B) frequency-based removal for lines that repeat a lot

    Operates line-by-line; best used BEFORE you join wrapped lines.
    """
    regexes = regexes or DEFAULT_HEADER_FOOTER_REGEXES
    protect_regexes = protect_regexes or PROTECT_LINE_REGEXES

    lines = text.split("\n")

    # Precompile patterns
    rx_list = [re.compile(pat, re.IGNORECASE) for pat in regexes]
    protect_list = [re.compile(pat, re.IGNORECASE) for pat in protect_regexes]

    def is_protected(line: str) -> bool:
        return any(rx.search(line) for rx in protect_list)

    def matches_header_footer_regex(line: str) -> bool:
        return any(rx.search(line) for rx in rx_list)

    # 1) Build frequency table (digit-masked)
    norm_counts = Counter()
    for ln in lines:
        raw = ln.strip()
        if not raw:
            continue
        if len(raw) > max_line_len:
            continue
        if is_protected(raw):
            continue
        norm_counts[_norm_line_for_count(raw)] += 1

    # Candidate “repeated” lines
    repeated_norms = {k for k, v in norm_counts.items() if v >= min_repeats}

    # 2) Filter lines
    kept: List[str] = []
    for ln in lines:
        raw = ln.rstrip()

        if not raw.strip():
            kept.append("")  # keep blank lines (paragraph breaks)
            continue

        if matches_header_footer_regex(raw):
            continue

        if len(raw.strip()) <= max_line_len and not is_protected(raw):
            if _norm_line_for_count(raw) in repeated_norms:
                continue

        kept.append(raw)

    # Clean up excessive blank lines introduced by removals
    out = "\n".join(kept)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out


# ----------------------------
# Core cleaning
# ----------------------------

def _remove_zero_width(s: str) -> str:
    for ch in ZERO_WIDTH_CHARS:
        s = s.replace(ch, "")
    return s


def _normalize_spaces(s: str) -> str:
    for ch in NBSP_CHARS:
        s = s.replace(ch, " ")
    s = s.replace("\u2028", "\n").replace("\u2029", "\n")  # unicode line separators
    return s


def _strip_mid_paragraph_page_numbers(lines: List[str], *, min_neighbor_len: int = 25, max_digits: int = 4) -> List[str]:
    kept: List[str] = []
    for i, line in enumerate(lines):
        t = line.strip()
        is_digits_only = t.isdigit() and 1 <= len(t) <= max_digits
        if is_digits_only:
            prev = lines[i - 1].strip() if i > 0 else ""
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
            # looks like it interrupts a paragraph
            if len(prev) >= min_neighbor_len and len(nxt) >= min_neighbor_len:
                continue
            # standalone page marker between blanks
            if prev == "" and nxt == "":
                continue
        kept.append(line)
    return kept


def _strip_page_number_before_section_symbol(lines: List[str]) -> List[str]:
    """
    Fix lines like:
      '11 § 160.102 ...'  -> '§ 160.102 ...'
      ' 3 § '            -> '§'
    """
    out: List[str] = []
    for ln in lines:
        ln2 = re.sub(r"^\s*\d+\s+(§\s*)", r"\1", ln)
        out.append(ln2)
    return out


def clean_legal_text(value: Any, *, keep_paragraphs: bool = True) -> str:
    if value is None:
        return ""

    s = value if isinstance(value, str) else str(value)

    # 1) Unicode normalize (fix ligatures/compat chars)
    s = unicodedata.normalize("NFKC", s)

    # 2) Remove soft hyphen (often invisible) + zero-width junk + NBSPs
    s = s.replace("\u00ad", "")
    s = _remove_zero_width(s)
    s = _normalize_spaces(s)

    # 3) Decode literal escapes inside strings (\\n -> \n)
    s = s.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n").replace("\\t", " ")

    # 4) Normalize newline styles
    s = s.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")

    # 5) Trim trailing spaces per line
    s = "\n".join(line.rstrip() for line in s.split("\n"))

    # ---- line-based cleanup (best done BEFORE joining wrapped lines) ----
    lines = s.split("\n")
    lines = _strip_mid_paragraph_page_numbers(lines)
    lines = _strip_page_number_before_section_symbol(lines)
    s = "\n".join(lines)

    # 6) Collapse excessive blank lines (keep paragraphs)
    s = re.sub(r"\n{3,}", "\n\n", s)

    # 7) Fix "§" split across newline: '§\n 160.400' -> '§ 160.400'
    s = re.sub(r"§\s*\n\s*", "§ ", s)

    # 8) Dehyphenate word breaks across newlines: 'super-\nvisory' -> 'supervisory'
    s = re.sub(r"(\w)-\n(\w)", r"\1\2", s)

    # 9) Fix in-word newline splits (VERY common in your example):
    #    'Re\ngulation' -> 'Regulation', 'PROVID\nERS' -> 'PROVIDERS'
    s = re.sub(r"([A-Za-z])\n([A-Za-z])", r"\1\2", s)

    # 10) Fix split numbers across newlines:
    #     '§ 1\n60.103' -> '§ 160.103', '16\n4.524' -> '164.524'
    s = re.sub(r"(\d)\n(\d)", r"\1\2", s)

    # 11) Fix split last-letter artifacts like 'entit y' -> 'entity', 'welfar e' -> 'welfare'
    #     Only merges if the last letter is lowercase, so 'Part A' stays as-is.
    s = re.sub(r"\b([A-Za-z]{5,})\s([a-z])\b", r"\1\2", s)

    # 12) Now join wrapped lines *within* paragraphs (preserve paragraph breaks)
    if keep_paragraphs:
        s = re.sub(r"(?<!\n)\n(?!\n)", " ", s)   # single newline -> space
        s = re.sub(r"\n{3,}", "\n\n", s)
    else:
        s = re.sub(r"\n+", " ", s)

    # 13) Normalize spacing
    s = re.sub(r"[ \t]{2,}", " ", s)

    return s.strip()


# ----------------------------
# Directory clean/write
# ----------------------------

def clean_json_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    pattern: str = "*.json",
    overwrite: bool = True,
    keep_paragraphs: bool = True,
    remove_headers_footers: bool = True,
    min_header_footer_repeats: int = 4,
) -> Dict[str, int]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    errors = 0

    for src_path in sorted(input_dir.glob(pattern)):
        if not src_path.is_file():
            continue

        dst_path = output_dir / src_path.name
        if dst_path.exists() and not overwrite:
            skipped += 1
            continue

        try:
            with src_path.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            if not isinstance(obj, dict):
                raise ValueError("JSON root must be an object/dict.")

            title = obj.get("title")
            if not isinstance(title, str) or not title.strip():
                obj["title"] = src_path.stem

            obj["all_text"] = clean_legal_text(obj.get("all_text"), keep_paragraphs=True)


            with dst_path.open("w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)

            written += 1

        except Exception as e:
            errors += 1
            print(f"Error processing {src_path.name}: {e}")

    return {"written": written, "skipped": skipped, "errors": errors}


if __name__ == "__main__":
    summary = clean_json_directory(
        "./mock_data/raw_data",
        "./mock_data/cleaned_data",
        keep_paragraphs=True,            # best for legal text + LLM prompts
        remove_headers_footers=True,     # enable header/footer stripping
        min_header_footer_repeats=4,     # bump to 6–10 if it removes too much
        overwrite=True,
    )
    print(summary)
