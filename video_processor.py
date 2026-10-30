from pathlib import Path
from collections import Counter
import cv2

def process_video(model, input_path: str, output_path: str, conf=0.35, stride=2, max_width=640, progress=None):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("OpenCV could not open the uploaded video.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, max_width / max(width, 1))
    out_w, out_h = max(2, int(width*scale)), max(2, int(height*scale))
    out_w -= out_w % 2
    out_h -= out_h % 2
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w,out_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create MP4 output. Check the output folder and codec support.")
    frame_index = 0
    processed = 0
    total_detections = 0
    class_counts = Counter()
    last_annotated = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if scale != 1.0:
                frame = cv2.resize(frame, (out_w,out_h))
            if frame_index % max(1,stride) == 0 or last_annotated is None:
                result = model.predict(source=frame, conf=conf, imgsz=640, verbose=False)[0]
                last_annotated = result.plot()
                if result.boxes is not None:
                    for cls_id in result.boxes.cls.cpu().tolist():
                        name = result.names.get(int(cls_id), str(int(cls_id))) if isinstance(result.names,dict) else result.names[int(cls_id)]
                        class_counts[str(name)] += 1
                        total_detections += 1
                processed += 1
            writer.write(last_annotated if last_annotated is not None else frame)
            frame_index += 1
            if progress is not None and total_frames:
                progress.progress(min(frame_index/total_frames,1.0), text=f"Processing frame {frame_index:,} / {total_frames:,}")
    finally:
        cap.release()
        writer.release()
    return {"frames_processed": processed, "total_detections": total_detections,
            "fps": fps, "class_counts": dict(class_counts), "total_frames": frame_index}
