
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from rfdetr import RFDETRSmall


def load_detector(model_path: str):
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(f"Model weights not found: {path}")

    return RFDETRSmall(
        pretrain_weights=str(path),
        device="cpu",
        trust_checkpoint=True
    )


def detect_image(model, image, conf: float = 0.35, imgsz: int = 640):
    """Run local RF-DETR inference on a PIL RGB image."""

    if isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    else:
        pil_image = Image.fromarray(np.asarray(image)).convert("RGB")

    detections = model.predict(
        pil_image,
        threshold=conf
    )

    annotated = pil_image.copy()
    draw = ImageDraw.Draw(annotated)

    rows = []

    boxes = detections.xyxy
    scores = detections.confidence
    class_ids = detections.class_id
    class_names = detections.data.get("class_name")

    for i, (box, score, class_id) in enumerate(
        zip(boxes, scores, class_ids)
    ):
        x1, y1, x2, y2 = map(float, box)

        if class_names is not None:
            name = str(class_names[i])
        else:
            name = f"class_{int(class_id)}"

        score = float(score)

        draw.rectangle(
            (x1, y1, x2, y2),
            outline="lime",
            width=3
        )

        draw.text(
            (x1, max(0, y1 - 18)),
            f"{name} {score:.2f}",
            fill="lime"
        )

        rows.append({
            "class_name": name,
            "confidence": round(score, 4),
            "x1": round(x1, 1),
            "y1": round(y1, 1),
            "x2": round(x2, 1),
            "y2": round(y2, 1),
            "center_x": round((x1 + x2) / 2, 1),
            "center_y": round((y1 + y2) / 2, 1),
        })

    annotated_rgb = np.array(annotated)

    dataframe = pd.DataFrame(rows, columns=[
        "class_name", "confidence",
        "x1", "y1", "x2", "y2",
        "center_x", "center_y"
    ])

    return annotated_rgb, dataframe


def model_info(model_path: str):
    path = Path(model_path)

    return {
        "model_path": str(path),
        "weights_found": path.exists(),
        "file_size_mb": round(
            path.stat().st_size / (1024 * 1024), 2
        ) if path.exists() else None,
        "inference_mode": "Local / on-device",
        "hosted_api_required": False,
    }
