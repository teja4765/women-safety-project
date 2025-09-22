"""
Image analysis API endpoints (single-frame threat check)
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import numpy as np
import cv2
import os
import uuid
import logging

from app.services.detection import PersonDetector
from app.services.gender_classifier import GenderClassifierService
from typing import Optional

logger = logging.getLogger(__name__)

router = APIRouter()

# Lazy singleton for detector
_detector: Optional[PersonDetector] = None
_gender_svc: Optional[GenderClassifierService] = None


async def get_detector() -> PersonDetector:
    global _detector
    if _detector is None:
        _detector = PersonDetector()
        await _detector.initialize()
    return _detector


async def get_gender_service() -> GenderClassifierService:
    global _gender_svc
    if _gender_svc is None:
        _gender_svc = GenderClassifierService()
        await _gender_svc.initialize()
    return _gender_svc

@router.post("/image-analysis/upload")
async def analyze_image(file: UploadFile = File(...)):
    """Upload an image (jpg/png), run person detection, return annotated image and JSON summary."""
    try:
        if not file.content_type or not any(
            file.content_type.lower().startswith(t) for t in ["image/jpeg", "image/png"]
        ):
            raise HTTPException(status_code=400, detail="File must be an image (jpg/png)")

        # Read bytes and decode with OpenCV
        data = await file.read()
        nparr = np.frombuffer(data, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        # Detect
        detector = await get_detector()
        detections = await detector.detect(bgr)
        gender_service = await get_gender_service()

        # Annotate and classify gender per detection
        annotated = bgr.copy()
        people = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"person {conf:.2f}"
            # Crop person and classify gender probability (female probability)
            crop = bgr[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            female_prob: Optional[float] = None
            if crop.size > 0:
                try:
                    female_prob = await gender_service.classify(crop)
                except Exception:
                    female_prob = None
            gender_label = ""
            if female_prob is not None:
                gender_label = f" | female:{female_prob:.2f} male:{1.0 - female_prob:.2f}"
            cv2.putText(
                annotated,
                label + gender_label,
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            people.append({
                "bbox": [x1, y1, x2, y2],
                "detection_confidence": float(conf),
                "female_probability": float(female_prob) if female_prob is not None else None,
                "male_probability": float(1.0 - female_prob) if female_prob is not None else None,
            })

        total_people = len(detections)

        # Placeholder attack probability (0..1). You can replace with a proper model.
        # Simple heuristic: more males than females increases risk slightly.
        num_females = sum(1 for p in people if p.get("female_probability") is not None and p["female_probability"] >= 0.5)
        num_males = sum(1 for p in people if p.get("female_probability") is not None and p["female_probability"] < 0.5)
        attack_probability = 0.0
        if total_people > 0:
            male_ratio = num_males / max(1, (num_males + num_females))
            attack_probability = float(min(1.0, max(0.0, 0.15 + 0.5 * male_ratio))) if num_males > num_females else 0.1
        threat = attack_probability >= 0.6
        threat_reason = "heuristic_threshold" if threat else ("no_people_detected" if total_people == 0 else "below_threshold")

        # Persist annotated image
        job_id = str(uuid.uuid4())
        out_dir = os.path.join("data", "image_analysis", job_id)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "annotated.jpg")
        cv2.imwrite(out_path, annotated)

        return {
            "job_id": job_id,
            "total_people": total_people,
            "people": people,
            "attack_probability": attack_probability,
            "threat": threat,
            "reason": threat_reason,
            "annotated_image_url": f"/api/v1/image-analysis/{job_id}/image",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in analyze_image: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/image-analysis/{job_id}/image")
async def get_annotated_image(job_id: str):
    try:
        out_path = os.path.join("data", "image_analysis", job_id, "annotated.jpg")
        if not os.path.exists(out_path):
            raise HTTPException(status_code=404, detail="Image not found")
        return FileResponse(out_path, media_type="image/jpeg")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving annotated image {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


