"""
app.py  —  Flask web application for the Road Accident Detection System.

Routes:
  GET  /               → upload page
  POST /detect         → run detection, show results
  GET  /history        → view past detections
  POST /history/clear  → clear all detection history
  GET  /settings       → email & Twilio credentials settings
  POST /api/detect-frame → real-time single frame detection (base64)
"""

import os
import json
import base64
import numpy as np
import cv2
from pathlib  import Path
from datetime import datetime
from flask    import Flask, render_template, request, redirect, url_for, flash, jsonify

from detector import run_detection
from alerter  import send_alert


# ── .env helpers ──────────────────────────────────────────────────────────────
def _load_env_file(env_path: Path):
    """Read KEY=VALUE pairs from a .env file and inject into os.environ."""
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, value = line.partition('=')
            os.environ.setdefault(key.strip(), value.strip())


def _save_env_file(env_path: Path, updates: dict):
    """Write / update KEY=VALUE pairs in the .env file."""
    data = {}
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                k, _, v = line.partition('=')
                data[k.strip()] = v.strip()
    data.update({k: v for k, v in updates.items() if v})
    with open(env_path, 'w') as f:
        for k, v in data.items():
            f.write(f'{k}={v}\n')
    for k, v in data.items():
        os.environ[k] = v


# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = 'accident_detection_secret_key_change_in_production'

BASE_DIR     = Path(__file__).parent
UPLOAD_DIR   = BASE_DIR / 'static' / 'uploads'
HISTORY_FILE = BASE_DIR / 'detection_history.json'
ENV_FILE     = BASE_DIR / '.env'

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
_load_env_file(ENV_FILE)

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.mp4', '.avi', '.mov', '.mkv', '.wmv'}
MAX_FILE_MB        = 200


# ── Helpers ───────────────────────────────────────────────────────────────────
def _allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _save_history(result: dict):
    history = []
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE) as f:
            history = json.load(f)
    history.insert(0, result)
    history = history[:50]
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)


def _load_history() -> list:
    if not HISTORY_FILE.exists():
        return []
    with open(HISTORY_FILE) as f:
        return json.load(f)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        flash('No file selected.', 'error')
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        flash('No file selected.', 'error')
        return redirect(url_for('index'))

    if not _allowed(file.filename):
        flash(f'Unsupported file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}', 'error')
        return redirect(url_for('index'))

    safe_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    save_path = UPLOAD_DIR / safe_name
    file.save(str(save_path))

    size_mb = save_path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        save_path.unlink()
        flash(f'File too large ({size_mb:.1f} MB). Maximum: {MAX_FILE_MB} MB.', 'error')
        return redirect(url_for('index'))

    sample_every = int(request.form.get('sample_every', 10))
    alert_email  = request.form.get('alert_email', '').strip()
    alert_phone  = request.form.get('alert_phone', '').strip()
    result = run_detection(str(save_path), sample_every=sample_every)
    result['original_filename'] = file.filename

    if result['accident_detected']:
        alert_status = send_alert(result, alert_email=alert_email,
                                  alert_phone=alert_phone)
        result['alert_status'] = alert_status

    _save_history(result)
    return render_template('results.html', result=result)


@app.route('/history')
def history():
    records = _load_history()
    return render_template('history.html', records=records)


@app.route('/history/clear', methods=['POST'])
def clear_history():
    """Delete all detection history records."""
    try:
        if HISTORY_FILE.exists():
            HISTORY_FILE.unlink()
        flash('Detection history cleared successfully!', 'success')
    except Exception as e:
        flash(f'Failed to clear history: {e}', 'error')
    return redirect(url_for('history'))


@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if request.method == 'POST':
        updates = {}

        # Email settings
        sender   = request.form.get('sender',   '').strip()
        password = request.form.get('password', '').strip().replace(' ', '')
        if sender:   updates['ALERT_EMAIL_SENDER']   = sender
        if password: updates['ALERT_EMAIL_PASSWORD'] = password

        # Twilio settings
        twilio_sid   = request.form.get('twilio_sid',   '').strip()
        twilio_token = request.form.get('twilio_token', '').strip()
        twilio_phone = request.form.get('twilio_phone', '').strip()
        twilio_wa    = request.form.get('twilio_whatsapp', '').strip()
        if twilio_sid:   updates['TWILIO_ACCOUNT_SID']    = twilio_sid
        if twilio_token: updates['TWILIO_AUTH_TOKEN']      = twilio_token
        if twilio_phone: updates['TWILIO_PHONE_NUMBER']    = twilio_phone
        if twilio_wa:    updates['TWILIO_WHATSAPP_FROM']   = twilio_wa

        if updates:
            _save_env_file(ENV_FILE, updates)
            flash('Credentials saved successfully!', 'success')
        else:
            flash('Nothing to save — fields were blank.', 'error')
        return redirect(url_for('settings'))

    cfg = {
        'sender'          : os.environ.get('ALERT_EMAIL_SENDER',   ''),
        'password'        : os.environ.get('ALERT_EMAIL_PASSWORD', ''),
        'twilio_sid'      : os.environ.get('TWILIO_ACCOUNT_SID',   ''),
        'twilio_token'    : os.environ.get('TWILIO_AUTH_TOKEN',    ''),
        'twilio_phone'    : os.environ.get('TWILIO_PHONE_NUMBER',  ''),
        'twilio_whatsapp' : os.environ.get('TWILIO_WHATSAPP_FROM', ''),
    }
    return render_template('settings.html', cfg=cfg)


@app.route('/api/result/<timestamp>')
def api_result(timestamp):
    records = _load_history()
    for r in records:
        if r.get('timestamp') == timestamp:
            return jsonify(r)
    return jsonify({'error': 'Not found'}), 404


@app.route('/api/detect-frame', methods=['POST'])
def api_detect_frame():
    """
    Accept a base64-encoded image frame and return detection result.
    Used by the real-time webcam detection panel and the frame-by-frame visualizer.
    """
    try:
        from detector import _load, _predict

        data = request.get_json()
        if not data or 'frame' not in data:
            return jsonify({'error': 'No frame data'}), 400

        # Decode base64 image
        img_b64 = data['frame']
        # Strip data URI prefix if present
        if ',' in img_b64:
            img_b64 = img_b64.split(',', 1)[1]

        img_bytes = base64.b64decode(img_b64)
        nparr     = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Could not decode frame'}), 400

        model, config = _load()
        is_accident, probability = _predict(model, config, frame)

        # Check if client wants to save accident frames
        save_accident = data.get('save_accident_frame', False)
        timestamp = data.get('timestamp')
        frame_no = data.get('frame_no', 0)
        saved_path = None

        if is_accident and save_accident and timestamp:
            from detector import _annotate, RESULTS_DIR
            run_dir = RESULTS_DIR / timestamp
            run_dir.mkdir(parents=True, exist_ok=True)
            
            # Annotate and save on the server
            annotated = _annotate(frame, is_accident, probability, frame_no)
            out_name = f'frame_{frame_no:06d}.jpg'
            out_path = run_dir / out_name
            cv2.imwrite(str(out_path), annotated)
            saved_path = f'results/{timestamp}/{out_name}'

        return jsonify({
            'is_accident': is_accident,
            'probability': round(float(probability), 4),
            'saved_path': saved_path
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/save-client-results', methods=['POST'])
def save_client_results():
    """
    Save the final results of a client-side frame-by-frame analysis run
    to the history database, and trigger notification alerts if needed.
    """
    try:
        result = request.get_json()
        if not result:
            return jsonify({'error': 'No data'}), 400

        # Format required keys
        result['original_filename'] = result.get('filename', 'video.mp4')
        result['status'] = 'accident_detected' if result.get('accident_detected') else 'no_accident'
        
        # Trigger alerts if accident was found
        if result.get('accident_detected'):
            alert_status = send_alert(
                result, 
                alert_email=result.get('alert_email', '').strip(),
                alert_phone=result.get('alert_phone', '').strip()
            )
            result['alert_status'] = alert_status

        # Save result to history
        _save_history(result)
        return jsonify({'success': True, 'timestamp': result.get('timestamp')})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/results/<timestamp>')
def results_detail(timestamp):
    """View details of a past detection run by timestamp."""
    records = _load_history()
    for r in records:
        if r.get('timestamp') == timestamp:
            return render_template('results.html', result=r)
    flash('Result record not found.', 'error')
    return redirect(url_for('history'))


@app.route('/history/delete/<timestamp>', methods=['POST'])
def delete_history_item(timestamp):
    """Delete a single detection history record and its associated directory."""
    try:
        records = _load_history()
        # Filter out the record
        new_records = [r for r in records if r.get('timestamp') != timestamp]
        
        # Save updated list
        with open(HISTORY_FILE, 'w') as f:
            json.dump(new_records, f, indent=2)

        # Delete corresponding results folder if it exists
        result_dir = BASE_DIR / 'static' / 'results' / timestamp
        if result_dir.exists():
            import shutil
            shutil.rmtree(str(result_dir))

        flash('History record deleted successfully!', 'success')
    except Exception as e:
        flash(f'Failed to delete history record: {e}', 'error')
    return redirect(url_for('history'))


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('=' * 55)
    print(' Road Accident Detection System')
    print(' Open http://127.0.0.1:5000 in your browser')
    print('=' * 55)
    app.run(debug=True, host='0.0.0.0', port=5000)