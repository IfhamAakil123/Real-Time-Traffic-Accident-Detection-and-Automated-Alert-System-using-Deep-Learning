# Road Accident Detection System
## Setup & Run Guide (Windows)

---

## FOLDER STRUCTURE
```
accident_project/
├── model/
│   ├── accident_model.h5       ← copy here after Colab training
│   ├── class_indices.json      ← copy here after Colab training
│   └── model_config.json       ← copy here after Colab training
├── static/
│   ├── uploads/                ← auto-created on first run
│   └── results/                ← auto-created on first run
├── templates/
│   ├── index.html
│   └── results.html
├── app.py
├── detector.py
├── alerter.py
├── requirements.txt
└── README.md
```

---

## STEP 1 — Install Python dependencies

Open CMD inside the `accident_project` folder and run:

```
pip install flask==3.0.3 tensorflow==2.15.0 opencv-python==4.9.0.80 numpy==1.26.4 Pillow==10.3.0 plyer==2.1.0
```

This takes 3–5 minutes. TensorFlow is the largest download (~500 MB).

---

## STEP 2 — Copy your trained model files

After running the Colab notebook, you will have downloaded:
- accident_model.h5
- class_indices.json
- model_config.json

Copy all three into the `model/` folder inside `accident_project/`.

---

## STEP 3 — Configure email alerts (optional)

Open `alerter.py` and fill in these three lines:

```python
SENDER_EMAIL    = "your_gmail@gmail.com"
SENDER_PASSWORD = "your_16char_app_password"
RECEIVER_EMAIL  = "where_alerts_go@gmail.com"
```

How to get a Gmail App Password:
1. Go to myaccount.google.com
2. Security → 2-Step Verification (enable if not already)
3. Security → App Passwords
4. Select app: Mail, device: Windows → Generate
5. Copy the 16-character password (no spaces) into SENDER_PASSWORD

If you skip this step the app still works — alerts will just be skipped.

---

## STEP 4 — Run the app

```
python app.py
```

Then open your browser and go to:
```
http://localhost:5000
```

---

## STEP 5 — Upload a video or image

1. Click "Drop file here" or drag a video onto the page
2. Click "Run Detection"
3. Wait 1–3 minutes (longer for longer videos)
4. Results page shows: thumbnail, confidence, frame numbers, alert status
5. Download the annotated video from the results page

---

## SUPPORTED FILE TYPES
Videos : MP4, AVI, MOV, MKV, WMV
Images : JPG, JPEG, PNG, BMP
Max size: 500 MB

---

## TROUBLESHOOTING

**"Model not loaded" error**
→ Make sure accident_model.h5 is inside the model/ folder

**"No module named tensorflow"**
→ Run: pip install tensorflow==2.15.0

**"No module named flask"**
→ Run: pip install flask

**Email not sending**
→ Check SENDER_PASSWORD is the App Password (not your Gmail password)
→ Check 2-Step Verification is enabled on your Google account

**Video processing is slow**
→ Normal — CPU inference takes ~1 second per analysed frame
→ A 1-minute video at 30fps = 180 analysed frames ≈ 3 minutes

**Desktop notification not showing**
→ Run: pip install plyer
→ Check Windows Focus Assist is not blocking notifications

---

## HOW DETECTION WORKS

1. Video is split into frames
2. Every 10th frame is sent to the MobileNetV2 model
3. Model returns an accident probability (0.0 to 1.0)
4. If probability >= 0.5 (threshold), frame is flagged as accident
5. If any frames are flagged, alerts fire automatically
6. Annotated video is saved with confidence overlay on each frame

To change the detection threshold, edit model_config.json:
```json
{ "threshold": 0.5 }
```
Lower = more sensitive (more false positives)
Higher = less sensitive (may miss some accidents)
