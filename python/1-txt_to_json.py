import argparse
import json
from pathlib import Path

def convert_directory(in_dir: Path, out_dir: Path, recursive: bool = False) -> int:
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern = "**/*.txt" if recursive else "*.txt"
    txt_files = sorted(in_dir.glob(pattern))

    converted = 0
    for txt_path in txt_files:
        title = txt_path.stem
        all_text = txt_path.read_text(encoding="utf-8", errors="replace")
        payload = {"title": title, "all_text": all_text}

        out_path = out_dir / f"{title}.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        converted += 1

    return converted

def main():
    p = argparse.ArgumentParser(description="Convert .txt files to .json with {title, all_text}")
    p.add_argument("--in-dir", required=True, help="Input directory containing .txt files")
    p.add_argument("--out-dir", required=True, help="Output directory for .json files")
    p.add_argument("--recursive", action="store_true", help="Also scan subfolders for .txt files")
    args = p.parse_args()

    n = convert_directory(Path(args.in_dir), Path(args.out_dir), recursive=args.recursive)
    print(f"Converted {n} file(s)")

if __name__ == "__main__":
    main()
