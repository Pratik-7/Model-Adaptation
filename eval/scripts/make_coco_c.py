import os, pathlib, random, json, argparse
from tqdm import tqdm
import numpy as np
import cv2
import albumentations as A

CORRUPTIONS = [
    "gaussian_noise","shot_noise","defocus_blur","motion_blur",
    "zoom_blur","fog","contrast","jpeg_compression",
]
SEVERITIES = [1,2,3,4,5]

def t_gaussian_noise(s):   return A.GaussNoise(var_limit={1:(5,10),2:(10,20),3:(20,40),4:(40,80),5:(80,120)}[s], mean=0, p=1)
def t_shot_noise(s):       return A.ISONoise(intensity={1:(0.02,0.04),2:(0.04,0.06),3:(0.06,0.08),4:(0.08,0.12),5:(0.12,0.16)}[s], color_shift=(0.01,0.03), p=1)
def t_defocus_blur(s):     return A.Defocus(radius={1:(2,3),2:(3,5),3:(5,7),4:(7,9),5:(9,11)}[s], alias_blur=(0.0,0.05), p=1)
def t_motion_blur(s):      return A.MotionBlur(blur_limit={1:(3,5),2:(5,9),3:(9,13),4:(13,17),5:(17,21)}[s], allow_shifted=True, p=1)
def t_zoom_blur(s):        return A.ZoomBlur(max_factor={1:(1.01,1.03),2:(1.03,1.06),3:(1.06,1.09),4:(1.09,1.12),5:(1.12,1.15)}[s], step_factor=(0.01,0.02), p=1)

# Albumentations 1.4.7 uses fog_coef_lower/upper (floats) + alpha_coef (float)
def t_fog(s):
    rng = {1:(0.10,0.20), 2:(0.20,0.30), 3:(0.30,0.40), 4:(0.40,0.50), 5:(0.50,0.60)}[s]
    alpha = {1:0.08, 2:0.09, 3:0.10, 4:0.11, 5:0.12}[s]
    return A.RandomFog(fog_coef_lower=rng[0], fog_coef_upper=rng[1], alpha_coef=alpha, p=1)

def t_contrast(s):         return A.RandomBrightnessContrast(brightness_limit=(0,0), contrast_limit={1:(0.1,0.2),2:(0.2,0.3),3:(0.3,0.4),4:(0.4,0.5),5:(0.5,0.6)}[s], p=1)

# Use ImageCompression for JPEG in Albumentations 1.4.7
def t_jpeg(s):
    ql = {1:85, 2:75, 3:60, 4:45, 5:25}[s]
    qu = {1:90, 2:85, 3:75, 4:60, 5:45}[s]
    return A.ImageCompression(compression_type="jpeg", quality_lower=ql, quality_upper=qu, p=1)

TFNS = {
    "gaussian_noise": t_gaussian_noise,
    "shot_noise": t_shot_noise,
    "defocus_blur": t_defocus_blur,
    "motion_blur": t_motion_blur,
    "zoom_blur": t_zoom_blur,
    "fog": t_fog,
    "contrast": t_contrast,
    "jpeg_compression": t_jpeg,
}

def imread_rgb(p):
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None: raise RuntimeError(f"Failed to read {p}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def imwrite_rgb(p, img):
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", required=True, help="Folder with input .jpg images")
    ap.add_argument("--dst_root", required=True, help="Output root folder")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--limit", type=int, default=0, help="Process only first N images (0 = all)")
    ap.add_argument("--corruptions", nargs="*", default=CORRUPTIONS, choices=CORRUPTIONS)
    ap.add_argument("--severities", nargs="*", type=int, default=SEVERITIES, choices=SEVERITIES)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    SRC_DIR = pathlib.Path(args.src_dir); DST_ROOT = pathlib.Path(args.dst_root)

    images = sorted(SRC_DIR.glob("*.jpg"))
    if args.limit > 0: images = images[:args.limit]
    if not images: raise SystemExit(f"[ERROR] No .jpg images in {SRC_DIR}")

    for corr in args.corruptions:
        transforms = {s: TFNS[corr](s) for s in args.severities}
        for s in args.severities:
            (DST_ROOT / corr / f"severity_{s}").mkdir(parents=True, exist_ok=True)
        for ip in tqdm(images, desc=f"{corr:>17}"):
            img = imread_rgb(ip)
            for s in args.severities:
                outp = DST_ROOT / corr / f"severity_{s}" / ip.name
                if outp.exists(): continue
                aug = transforms[s](image=img)["image"]
                imwrite_rgb(outp, aug)

    (DST_ROOT / "manifest.json").write_text(json.dumps({
        "source_dir": str(SRC_DIR),
        "dst_root": str(DST_ROOT),
        "corruptions": args.corruptions,
        "severities": args.severities,
        "seed": args.seed,
        "albumentations": A.__version__,
        "opencv": cv2.__version__,
        "num_images": len(images),
        "limit": args.limit,
    }, indent=2))
    print("✅ COCO-C generated at", DST_ROOT)

if __name__ == "__main__":
    main()