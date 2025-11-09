import argparse, pathlib, json
import cv2
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", required=True, help="Folder with input .jpg images")
    ap.add_argument("--dst_root", required=True, help="Output root folder (will create jpeg_compression/severity_X)")
    ap.add_argument("--limit", type=int, default=0, help="Process only first N images (0 = all)")
    args = ap.parse_args()

    src = pathlib.Path(args.src_dir)
    dst_root = pathlib.Path(args.dst_root) / "jpeg_compression"

    # Severity → (quality_lower, quality_upper) — we’ll just use the lower bound for determinism.
    qualities = {
        1: (85, 90),
        2: (75, 85),
        3: (60, 75),
        4: (45, 60),
        5: (25, 45),
    }

    # Gather images
    imgs = sorted(src.glob("*.jpg"))
    if args.limit > 0:
        imgs = imgs[:args.limit]
    if not imgs:
        raise SystemExit(f"[ERROR] No .jpg files in {src}")

    # Create severity folders
    for sev in range(1, 6):
        (dst_root / f"severity_{sev}").mkdir(parents=True, exist_ok=True)

    # Write JPEGs with chosen quality
    for sev in range(1, 6):
        q = qualities[sev][0]  # pick deterministic lower bound
        params = [int(cv2.IMWRITE_JPEG_QUALITY), q]
        out_dir = dst_root / f"severity_{sev}"
        for ip in tqdm(imgs, desc=f"jpeg_compression sev={sev} (q={q})"):
            op = out_dir / ip.name
            if op.exists():
                continue
            img = cv2.imread(str(ip), cv2.IMREAD_COLOR)
            if img is None:
                continue
            cv2.imwrite(str(op), img, params)

    # Manifest
    (dst_root / "manifest_jpeg.json").write_text(json.dumps({
        "source_dir": str(src),
        "dst_root": str(dst_root),
        "qualities": {k: v[0] for k, v in qualities.items()},
        "num_images": len(imgs),
        "limit": args.limit,
        "backend": f"opencv {cv2.__version__}"
    }, indent=2))
    print("✅ JPEG corruption generated at", dst_root)

if __name__ == "__main__":
    main()