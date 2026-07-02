"""
detector.py
-----------
Loads the trained MobileNetV2 model and runs accident detection
on an uploaded video or image file.
"""

import cv2
import json
import numpy as np
from pathlib import Path
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import load_model

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
MODEL_PATH  = BASE_DIR / 'model' / 'accident_model.h5'
CONFIG_PATH = BASE_DIR / 'model' / 'model_config.json'
RESULTS_DIR = BASE_DIR / 'static' / 'results'

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Load model once at startup ─────────────────────────────────────────────────
_model  = None
_config = None

def _load():
    global _model, _config
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f'Model not found at {MODEL_PATH}\n'
                'Place accident_model.h5 inside the model/ folder.'
            )
        _model = load_model(str(MODEL_PATH))
        with open(CONFIG_PATH) as f:
            _config = json.load(f)
        print(f'[Detector] Model loaded. Classes: {_config["class_indices"]}')
    return _model, _config


# ── Helpers ───────────────────────────────────────────────────────────────────
def _preprocess(frame_bgr, img_size=(224, 224)):
    rgb     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, img_size)
    return resized.astype(np.float32) / 255.0


def _predict(model, config, frame_bgr):
    processed = _preprocess(frame_bgr, tuple(config['img_size']))
    batch     = np.expand_dims(processed, 0)
    raw_prob  = float(model.predict(batch, verbose=0)[0][0])
    acc_idx   = config['class_indices'].get('accident', 1)
    acc_prob  = raw_prob if acc_idx == 1 else (1.0 - raw_prob)
    threshold = config.get('threshold', 0.5)
    return acc_prob >= threshold, acc_prob


def _annotate(frame_bgr, is_accident, probability, frame_no):
    out   = frame_bgr.copy()
    h, w  = out.shape[:2]
    label = f'ACCIDENT  {probability:.1%}' if is_accident else f'Normal  {1-probability:.1%}'
    color = (0, 0, 220) if is_accident else (0, 180, 0)

    cv2.rectangle(out, (0, 0), (w, 46), (0, 0, 0), -1)
    cv2.rectangle(out, (0, 0), (w, 46), color, 2)
    cv2.putText(out, label, (12, 30), cv2.FONT_HERSHEY_DUPLEX,
                0.85, color, 2, cv2.LINE_AA)
    cv2.putText(out, f'Frame {frame_no}', (w - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
    if is_accident:
        cv2.rectangle(out, (2, 2), (w - 2, h - 2), (0, 0, 220), 3)
    return out


# ── Public API ────────────────────────────────────────────────────────────────
def run_detection(file_path: str, sample_every: int = 10) -> dict:
    """
    Run detection on an image or video.
    Returns a result dict consumed by app.py and alerter.py.
    """
    model, config = _load()
    file_path     = Path(file_path)
    suffix        = file_path.suffix.lower()
    timestamp     = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir       = RESULTS_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    result = {
        'status'               : 'no_accident',
        'accident_detected'    : False,
        'max_confidence'       : 0.0,
        'total_frames'         : 0,
        'flagged_frames'       : 0,
        'flagged_images'       : [],
        'first_accident_frame' : None,
        'best_evidence_frame'  : None,
        'duration_seconds'     : None,
        'timestamp'            : timestamp,
        'filename'             : file_path.name,
        'error'                : None,
    }

    try:
        # ── IMAGE ──────────────────────────────────────────────────────────────
        if suffix in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
            frame = cv2.imread(str(file_path))
            if frame is None:
                raise ValueError('Could not read image file.')

            is_acc, prob = _predict(model, config, frame)
            annotated    = _annotate(frame, is_acc, prob, 1)
            out_path     = run_dir / 'frame_001.jpg'
            cv2.imwrite(str(out_path), annotated)

            result['total_frames']   = 1
            result['max_confidence'] = prob

            if is_acc:
                evidence_path = f'results/{timestamp}/frame_001.jpg'
                result.update({
                    'accident_detected'    : True,
                    'status'               : 'accident_detected',
                    'flagged_frames'       : 1,
                    'first_accident_frame' : 1,
                    'best_evidence_frame'  : evidence_path,
                })
                result['flagged_images'].append(evidence_path)

        # ── VIDEO ──────────────────────────────────────────────────────────────
        elif suffix in {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}:
            cap = cv2.VideoCapture(str(file_path))
            if not cap.isOpened():
                raise ValueError('Could not open video file.')

            fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            result['duration_seconds'] = round(total / fps, 2) if total > 0 else None

            frame_no      = 0
            saved_no      = 0
            max_saved     = 12
            best_acc_prob = 0.0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_no += 1
                result['total_frames'] = frame_no

                if frame_no % sample_every != 0:
                    continue

                is_acc, prob = _predict(model, config, frame)
                if prob > result['max_confidence']:
                    result['max_confidence'] = prob

                # Save all sampled frames as annotated thumbnails
                annotated = _annotate(frame, is_acc, prob, frame_no)
                out_name  = f'frame_{frame_no:06d}.jpg'
                cv2.imwrite(str(run_dir / out_name), annotated)

                if is_acc:
                    result['accident_detected'] = True
                    result['status']            = 'accident_detected'
                    result['flagged_frames']   += 1
                    if result['first_accident_frame'] is None:
                        result['first_accident_frame'] = frame_no
                    # Track highest-confidence flagged frame as best evidence
                    if prob > best_acc_prob:
                        best_acc_prob = prob
                        result['best_evidence_frame'] = f'results/{timestamp}/{out_name}'
                    if saved_no < max_saved:
                        result['flagged_images'].append(
                            f'results/{timestamp}/{out_name}'
                        )
                        saved_no += 1

            cap.release()

        else:
            raise ValueError(f'Unsupported file type: {suffix}')

        result['max_confidence'] = round(result['max_confidence'], 4)

    except Exception as e:
        result['status'] = 'error'
        result['error']  = str(e)
        print(f'[Detector] Error: {e}')

    return result
