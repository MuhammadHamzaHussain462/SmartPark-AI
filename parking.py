
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def analyze_grid(detections, image_width: int, image_height: int,
                 rows: int, cols: int):
    """Calculate numbered parking slot occupancy from vehicle detections."""

    rows = max(1, int(rows))
    cols = max(1, int(cols))

    occupied_cells = set()

    if detections is not None and not detections.empty:
        for _, item in detections.iterrows():
            x = float(item["center_x"])
            y = float(item["center_y"])

            if not (0 <= x < image_width and 0 <= y < image_height):
                continue

            col = min(cols - 1, int(x / image_width * cols))
            row = min(rows - 1, int(y / image_height * rows))

            occupied_cells.add((row, col))

    slot_rows = []

    for r in range(rows):
        states = {}

        for c in range(cols):
            slot_id = f"{chr(65 + r)}{c + 1}"
            state = (
                "Occupied"
                if (r, c) in occupied_cells
                else "Available"
            )
            states[slot_id] = state

        slot_rows.append({
            "Row": f"Row {r + 1}",
            **states
        })

    total = rows * cols
    occupied = len(occupied_cells)
    available = total - occupied

    return {
        "total": total,
        "occupied": occupied,
        "available": available,
        "occupancy_rate": occupied / total * 100,
        "occupied_cells": occupied_cells,
        "slot_rows": slot_rows,
    }


def draw_grid(image, occupancy, rows: int, cols: int):
    """Draw green available and red occupied numbered parking slots."""

    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas, "RGBA")

    width, height = canvas.size

    try:
        font = ImageFont.truetype("arial.ttf", max(12, width // 65))
    except OSError:
        font = ImageFont.load_default()

    for r in range(rows):
        for c in range(cols):
            x1 = int(c * width / cols)
            y1 = int(r * height / rows)
            x2 = int((c + 1) * width / cols)
            y2 = int((r + 1) * height / rows)

            occupied = (r, c) in occupancy["occupied_cells"]

            if occupied:
                fill = (220, 38, 38, 95)
                outline = (255, 60, 60, 255)
                status = "OCCUPIED"
            else:
                fill = (22, 163, 74, 85)
                outline = (34, 197, 94, 255)
                status = "AVAILABLE"

            draw.rectangle(
                (x1, y1, x2, y2),
                fill=fill,
                outline=outline,
                width=max(2, width // 350)
            )

            slot_id = f"{chr(65 + r)}{c + 1}"
            label = f"{slot_id} - {status}"

            draw.text(
                (x1 + 6, y1 + 6),
                label,
                font=font,
                fill=(255, 255, 255, 255),
                stroke_width=2,
                stroke_fill=(0, 0, 0, 220)
            )

    return canvas
