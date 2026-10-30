# SmartPark AI

A Streamlit computer-vision prototype for local vehicle detection, recorded-video analysis, and configurable parking-grid occupancy estimation.

## 1. Project structure

```text
SmartPark_AI/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── models/
│   └── best.pt                 # Add your trained Roboflow-exported weights here
├── src/
│   ├── __init__.py
│   ├── detector.py
│   ├── video_processor.py
│   ├── parking.py
│   └── analytics.py
└── assets/
    ├── samples/
    └── outputs/
```

## 2. Add the trained model

Copy your trained Ultralytics-compatible model weights into `models/` and name the file exactly:

`best.pt`

The application expects `SmartPark_AI/models/best.pt`. The model weights are not bundled in this source ZIP because model files are large and your screenshot only showed the filename, not the actual model bytes.

## 3. Windows setup

Open a terminal in this folder:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If Python 3.11 is not installed, use a supported Python 3.10 or 3.12 environment. Avoid Python 3.14 for this computer-vision stack unless all wheels are confirmed available.

## 4. Run

```powershell
streamlit run app.py
```

## 5. Features

- Local YOLO inference on uploaded images
- Bounding-box annotations and confidence scores
- Detection table and CSV export
- Recorded video processing and annotated MP4 download
- Configurable row/column grid occupancy estimate
- Model status and setup page

## 6. Important limitations

- The model must be trained for the vehicle classes you want to detect.
- The parking monitor currently divides the image into equal rectangular cells and assigns each detected object's center to a cell. This is an approximation, not calibrated parking-stall segmentation.
- Accurate stall occupancy requires a fixed camera and manually defined polygons for each parking bay, with perspective and occlusion testing.
- Video mode analyzes every Nth frame and reuses the last annotated frame between inference frames. Counts represent detections on analyzed frames, not unique vehicles.
- This application performs local inference and does not call a hosted Roboflow prediction API.

## 7. Troubleshooting

**Model not found:** confirm the file is `models/best.pt`, not `models/best.pt.pt`.

**Unsupported checkpoint:** export the model from Roboflow in an Ultralytics-compatible format. A dataset export is not the same as trained model weights.

**Blank page:** run `streamlit run app.py` from the project root and inspect the terminal traceback. Do not open `app.py` directly in a browser.

**Slow inference:** lower image size, reduce video processing width, or increase the frame stride.
