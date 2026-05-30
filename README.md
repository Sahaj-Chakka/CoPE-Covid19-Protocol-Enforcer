# CoPE — Covid19 Protocol Enforcer

> A real-time, multi-function surveillance system built on Raspberry Pi 4B to autonomously enforce COVID-19 safety protocols — reducing the need for human-to-human enforcement interaction.

---

## Overview

During the COVID-19 pandemic, frontline workers faced significant risk while enforcing safety protocols in public spaces. CoPE was built to minimize that human-to-human interaction without compromising enforcement effectiveness.

The system runs four independent detection modules in a continuous loop — all on a single Raspberry Pi 4B unit — and displays results in real time on a mini OLED screen.

---

## Features

| Module | Description |
|---|---|
| 😷 Face Mask Detector | Detects whether individuals in frame are wearing a mask using a CNN-based deep learning model |
| 🧑 Facial Recognition | Identifies registered individuals via `dlib` face encodings and logs digital attendance via email |
| 🌡️ Body Temperature Monitor | Non-invasive IR temperature reading averaged over 5 samples, displayed on OLED |
| 📏 Social Distance Detector | Uses YOLOv3 + OpenCV to detect bounding boxes around humans and flags proximity violations |

---

## Hardware

- Raspberry Pi 4B (4GB RAM)
- USB Webcam (HD)
- Infrared (IR) Temperature Sensor — I2C interface
- Ultrasonic Distance Sensor (HC-SR04)
- Mini OLED Display — I2C interface

---

## How It Works

The system operates in two modes:

**Mode 1 (Idle):** Displays live weather information on the OLED screen while the ultrasonic sensor continuously monitors for nearby presence.

**Mode 2 (Active):** Triggered when the ultrasonic sensor detects someone approaching. Sequentially runs:
1. Face Mask Detection
2. Body Temperature Measurement
3. Facial Recognition + Attendance Logging
4. Social Distance Detection (parallel monitoring)

After each cycle completes, the system resets and returns to Mode 1.

---

## Project Structure

```
CoPE/
├── startExecution.py              # Main entry point — orchestrates all modules
├── CoPE_enforcer.py               # Core enforcement logic
├── displayWeather.py              # Weather display for idle mode
├── requirements.txt
├── modules/
│   ├── maskDetection_module/
│   │   ├── mask_detector.py
│   │   ├── face_detector_models/  # Pre-trained CNN face detection model
│   │   └── mask_detector_models/  # Pre-trained mask classification model
│   ├── socialDistancing_module/
│   │   ├── socialDistancing_detector.py
│   │   └── yolov3/                # YOLOv3 weights, config, and COCO labels
│   ├── temperature_module/
│   │   └── temperature.py
│   ├── distanceMeasurer_module/
│   │   └── ultraSonicSensor.py
│   ├── display_module/
│   │   └── display.py
│   └── weatherService_module/
│       └── getWeather.py
```

---

## Tech Stack

- **Language:** Python 3
- **Hardware:** Raspberry Pi 4B, IR Sensor, Ultrasonic Sensor, OLED, USB Webcam
- **Libraries:** OpenCV, dlib, NumPy, smtplib, Adafruit (OLED), RPi.GPIO
- **ML Models:** Pre-trained CNN (face detection), custom mask classifier, YOLOv3 (COCO)

---

## Setup & Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/Sahaj-Chakka/CoPE-Covid19-Protocol-Enforcer.git
cd CoPE-Covid19-Protocol-Enforcer
```

---

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

> Requires Python 3.7+. On Raspberry Pi, also enable I2C via `sudo raspi-config` → Interface Options → I2C → Enable.

---

### Step 3 — Download ML model files

These files are too large for GitHub and must be downloaded separately. Place each file in the folder shown below.

---

#### 🔹 Model 1 — YOLOv3 Weights (Social Distancing)

**Folder:** `modules/socialDistancing_module/yolov3/`

```bash
cd modules/socialDistancing_module/yolov3/

# Download weights (~237MB)
wget https://pjreddie.com/media/files/yolov3.weights

# Download config
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg

# Download COCO class labels
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
```

After this, your folder should contain:
```
yolov3/
├── yolov3.weights   ← 237MB
├── yolov3.cfg
└── coco.names
```

---

#### 🔹 Model 2 — Face Detection Caffe Model (Mask Detection)

**Folder:** `modules/maskDetection_module/face_detector_models/`

```bash
cd modules/maskDetection_module/face_detector_models/

# Download the prototxt (model architecture)
wget https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt

# Download the caffemodel (pre-trained weights, ~10MB)
wget https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
```

After this, your folder should contain:
```
face_detector_models/
├── deploy.prototxt
└── res10_300x300_ssd_iter_140000.caffemodel
```

---

#### 🔹 Model 3 — Mask Classifier Model

**Folder:** `modules/maskDetection_module/mask_detector_models/`

Download the pre-trained mask detection model from this open-source project:

👉 [https://github.com/chandrikadeb7/Face-Mask-Detection](https://github.com/chandrikadeb7/Face-Mask-Detection)

1. Go to the link above
2. Click **Code** → **Download ZIP**
3. Unzip it → find the file `mask_detector.model`
4. Rename it to `pretrained1.model`
5. Place it in `modules/maskDetection_module/mask_detector_models/`

After this, your folder should contain:
```
mask_detector_models/
└── pretrained1.model
```

---

### Step 4 — Configure environment variables

Create a `.env` file in the root folder (or set these in your shell):

```bash
# Weather API — get a free key at https://openweathermap.org/api
export OPENWEATHER_API_KEY="your_api_key_here"
export COPE_CITY="Agartala"

# Email for attendance notifications
export COPE_EMAIL_SENDER="your_gmail@gmail.com"
export COPE_EMAIL_PASSWORD="your_app_password"
export COPE_EMAIL_RECEIVER="admin@example.com"
```

> For Gmail, use an **App Password** (not your regular password). Generate one at: Google Account → Security → 2-Step Verification → App Passwords.

---

### Step 5 — Run the system

```bash
python startExecution.py
```

The system will start in **Mode 1 (Idle)** — displaying weather on the OLED and watching for nearby presence via the ultrasonic sensor. When someone approaches, it automatically switches to **Mode 2 (Active)** and runs all four detection modules.

Press `Q` on any OpenCV window or `Ctrl+C` in Terminal to stop.

---

## Team

Built as a Final Year Project by Group 4, under the guidance of **Dr. Arvind Kumar Jain**.

- Ayush Ranjan Singh
- Priyam Chatterjee
- Sukrit Raj
- Sahaj Chakka
- Ved Prakash
