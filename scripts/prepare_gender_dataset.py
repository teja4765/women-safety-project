import argparse
import asyncio
import os
from typing import List, Optional

import cv2
import numpy as np

from app.services.detection import PersonDetector
from app.services.gender_classifier import GenderClassifierService


def list_images(root: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    paths: List[str] = []
    for base, _, files in os.walk(root):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in exts:
                paths.append(os.path.join(base, f))
    return paths


async def process_dataset(src_dir: str, out_dir: str, use_auto_label: bool) -> None:
    os.makedirs(out_dir, exist_ok=True)
    male_dir = os.path.join(out_dir, "male")
    female_dir = os.path.join(out_dir, "female")
    unknown_dir = os.path.join(out_dir, "unknown")
    for d in [male_dir, female_dir, unknown_dir]:
        os.makedirs(d, exist_ok=True)

    detector = PersonDetector()
    await detector.initialize()

    gender_svc: Optional[GenderClassifierService] = None
    if use_auto_label:
        gender_svc = GenderClassifierService()
        await gender_svc.initialize()

    images = list_images(src_dir)
    saved = 0
    for idx, img_path in enumerate(images):
        bgr = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            continue

        detections = await detector.detect(bgr)
        if not detections:
            continue

        # Sort by area, largest first
        detections.sort(key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]), reverse=True)

        for j, det in enumerate(detections):
            x1, y1, x2, y2 = det["bbox"]
            h, w = bgr.shape[:2]
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))
            if x2 - x1 < 16 or y2 - y1 < 16:
                continue
            crop = bgr[y1:y2, x1:x2]
            dest_dir = unknown_dir
            if gender_svc is not None:
                try:
                    female_p = await gender_svc.classify(crop)
                    if female_p >= 0.5:
                        dest_dir = female_dir
                    else:
                        dest_dir = male_dir
                except Exception:
                    dest_dir = unknown_dir

            out_name = f"{os.path.splitext(os.path.basename(img_path))[0]}_{j:02d}.jpg"
            out_path = os.path.join(dest_dir, out_name)
            # Use imwrite with paths that may contain special characters
            cv2.imencode('.jpg', crop)[1].tofile(out_path)
            saved += 1

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(images)} images, saved {saved} crops")

    print(f"Done. Saved {saved} person crops into: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare gender dataset by detecting people and optional auto-labeling")
    parser.add_argument("src", help="Source directory with mixed images")
    parser.add_argument("--out", default="data/prepared_gender_dataset", help="Output directory for prepared dataset")
    parser.add_argument("--auto-label", action="store_true", help="Auto-label male/female if a gender model is available")
    args = parser.parse_args()

    asyncio.run(process_dataset(args.src, args.out, args.auto_label))


if __name__ == "__main__":
    main()


