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

```bash
# Clone the repository
git clone https://github.com/your-username/CoPE.git
cd CoPE

# Install dependencies
pip install -r requirements.txt

# Run the system
python startExecution.py
```

> Requires Raspberry Pi OS with Python 3.7+. Ensure I2C is enabled via `raspi-config` for the OLED and IR sensor.

---

## Team

Built as a Final Year Project by Group 4, under the guidance of **Dr. Arvind Kumar Jain**.

- Ayush Ranjan Singh
- Sahaj Chakka
- Sukrit Raj
- Priyam Chatterjee
- Ved Prakash

