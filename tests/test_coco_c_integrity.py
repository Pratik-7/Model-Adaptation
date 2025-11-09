import sys, random, statistics, pathlib
from collections import defaultdict, Counter
import cv2
import numpy as np

SRC_DIR = pathlib.Path("data/coco/images/val2017")
DST_ROOT = pathlib.Path("data/coco_c")

CORRUPTIONS = [
    "gaussian_noise","shot_noise","defocus_blur","motion_blur",
    "zoom_blur","fog","contrast","jpeg_compression",
]
SEVERITIES = [1,2,3,4,5]

SAMPLE_N = 10            # files to spot-check per corruption/severity
DIFF_THRESH = 1.0        # mean absolute pixel diff threshold vs clean (0..255)

def fail(msg):
    print(f"❌ {msg}")
    sys.exit(1)

def ok(msg):
    print(f"✅ {msg}")

def imread_rgb(p):
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def median_file_size(paths):
    sizes = [p.stat().st_size for p in paths if p.is_file()]
    return statistics.median(sizes) if sizes else 0

def main():
    # 0) source images
    src_imgs = sorted(SRC_DIR.glob("*.jpg"))
    if not src_imgs:
        fail(f"No source images found at {SRC_DIR}")
    N = len(src_imgs)
    ok(f"Found {N} source images")
    name_to_src = {p.name: p for p in src_imgs}

    # 1) structure + counts
    totals = defaultdict(int)
    for corr in CORRUPTIONS:
        for s in SEVERITIES:
            folder = DST_ROOT / corr / f"severity_{s}"
            if not folder.exists():
                fail(f"Missing folder: {folder}")
            imgs = sorted(folder.glob("*.jpg"))
            totals[(corr, s)] = len(imgs)
            if len(imgs) != N:
                fail(f"Count mismatch for {corr}/severity_{s}: got {len(imgs)} expected {N}")
    ok("All corruption/severity folders present with correct counts")

    # 2) sanity: sample readability + non-identity vs clean
    rng = random.Random(123)
    for corr in CORRUPTIONS:
        for s in SEVERITIES:
            folder = DST_ROOT / corr / f"severity_{s}"
            imgs = sorted(folder.glob("*.jpg"))
            sample = rng.sample(imgs, min(SAMPLE_N, len(imgs)))
            diffs = []
            for op in sample:
                ip = name_to_src.get(op.name)
                if ip is None:
                    fail(f"Output {op} has no matching source name")
                src = imread_rgb(ip); dst = imread_rgb(op)
                if src is None or dst is None:
                    fail(f"Failed to read image (src or dst): {ip} / {op}")
                if src.shape != dst.shape:
                    fail(f"Shape mismatch for {op} vs {ip}: {dst.shape} vs {src.shape}")
                diff = np.mean(np.abs(dst.astype(np.float32) - src.astype(np.float32)))
                diffs.append(diff)
            # allow tiny diffs from rounding; require average diff above threshold
            if diffs and (sum(diffs)/len(diffs) < DIFF_THRESH):
                fail(f"Images in {corr}/severity_{s} look too similar to clean (mean diff {sum(diffs)/len(diffs):.3f})")
    ok("Random spot-checks: images readable and differ from clean")

    # 3) jpeg_compression: median file size should non-increase with severity
    jpeg_medians = []
    for s in SEVERITIES:
        folder = DST_ROOT / "jpeg_compression" / f"severity_{s}"
        imgs = sorted(folder.glob("*.jpg"))
        jpeg_medians.append((s, median_file_size(imgs)))
    non_increasing = all(jpeg_medians[i][1] >= jpeg_medians[i+1][1] for i in range(len(jpeg_medians)-1))
    if not non_increasing:
        fail(f"jpeg_compression median sizes not non-increasing: {jpeg_medians}")
    ok(f"jpeg_compression median sizes OK (non-increasing): {jpeg_medians}")

    # 4) filename coverage uniqueness
    for corr in CORRUPTIONS:
        names_sets = []
        for s in SEVERITIES:
            folder = DST_ROOT / corr / f"severity_{s}"
            names_sets.append({p.name for p in folder.glob('*.jpg')})
        common = set.intersection(*names_sets)
        if len(common) != N:
            fail(f"Filename mismatch across severities for {corr}")
    ok("Filenames align across severities for each corruption")

    # summary table
    print("\n=== Summary (images per folder) ===")
    for corr in CORRUPTIONS:
        row = [f"{corr:18}"] + [f"{totals[(corr,s)]:5d}" for s in SEVERITIES]
        print(" ".join(row))
    print("\nAll tests passed.")
    sys.exit(0)

if __name__ == "__main__":
    main()
