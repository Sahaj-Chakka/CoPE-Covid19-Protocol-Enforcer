"""
CoPE - Covid19 Protocol Enforcer
=================================
Main execution file. Orchestrates all modules in a sequential loop.

Mode 1 (Idle)  : Displays weather info. Ultrasonic sensor watches for presence.
Mode 2 (Active): Person detected → runs Mask Detection → Temperature →
                 Facial Recognition → Social Distance check.

Hardware: Raspberry Pi 4B, USB Webcam, IR Temp Sensor, Ultrasonic Sensor, OLED
Author  : Sahaj Chakka et al. — NIT Agartala, Group 4
"""

import time
import sys
import logging

# ── Module imports ────────────────────────────────────────────────────────────
from modules.distanceMeasurer_module.ultraSonicSensor import UltrasonicSensor
from modules.maskDetection_module.mask_detector       import MaskDetector
from modules.temperature_module.temperature           import TemperatureSensor
from modules.socialDistancing_module.socialDistancing_detector import SocialDistancingDetector
from modules.display_module.display                   import OLEDDisplay
from modules.weatherService_module.getWeather         import get_weather
from displayWeather                                   import display_weather_screen

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("CoPE")

# ── Configuration ─────────────────────────────────────────────────────────────
PROXIMITY_THRESHOLD_CM = 150   # Trigger active mode if person within 150 cm
ACTIVE_MODE_DURATION_S = 10    # How long each module runs (seconds)
CYCLE_PAUSE_S          = 2     # Pause between cycles


def run_active_mode(mask_detector, temp_sensor, social_detector, display):
    """
    Runs all enforcement modules in sequence when a person is detected.
    """
    log.info("=== ACTIVE MODE TRIGGERED ===")

    # ── Step 1: Mask Detection ─────────────────────────────────────────────
    log.info("Running Mask Detection...")
    display.show_message("Detecting mask...", line2="Please look at camera")
    mask_result = mask_detector.run(duration_seconds=ACTIVE_MODE_DURATION_S)
    log.info(f"Mask result: {mask_result}")

    if mask_result["status"] == "NO_MASK":
        display.show_message("⚠ No Mask!", line2="Please wear a mask")
        time.sleep(2)
    else:
        display.show_message("✓ Mask OK", line2="Thank you")
        time.sleep(1)

    # ── Step 2: Temperature Measurement ───────────────────────────────────
    log.info("Measuring temperature...")
    display.show_message("Measuring temp...", line2="Stay still please")
    temp_c = temp_sensor.read_average_temperature(readings=5)
    log.info(f"Temperature: {temp_c:.1f}°C")

    if temp_c > 37.5:
        display.show_message(f"Temp: {temp_c:.1f}C", line2="⚠ High Temp!")
        log.warning(f"High temperature detected: {temp_c:.1f}C")
    else:
        display.show_message(f"Temp: {temp_c:.1f}C", line2="Normal ✓")
    time.sleep(2)

    # ── Step 3: Facial Recognition (Attendance) ───────────────────────────
    log.info("Running Facial Recognition...")
    display.show_message("Identifying...", line2="Face the camera")
    # Facial recognition is handled inside mask_detector (same camera feed)
    face_result = mask_detector.run_recognition(duration_seconds=ACTIVE_MODE_DURATION_S)
    if face_result["identified"]:
        display.show_message(f"Hello {face_result['name']}!", line2="Attendance logged ✓")
        log.info(f"Identified: {face_result['name']}")
    else:
        display.show_message("Unknown person", line2="Not in database")
    time.sleep(2)

    # ── Step 4: Social Distancing Check ───────────────────────────────────
    log.info("Running Social Distancing Detector...")
    display.show_message("Checking distance", line2="Stand apart please")
    social_result = social_detector.run(duration_seconds=ACTIVE_MODE_DURATION_S)
    if social_result["violation"]:
        display.show_message("⚠ Too Close!", line2="Maintain 6ft dist.")
        log.warning("Social distancing violation detected!")
    else:
        display.show_message("Distance OK ✓", line2="Stay safe!")
    time.sleep(2)

    log.info("=== ACTIVE MODE COMPLETE ===")


def main():
    log.info("Initializing CoPE — Covid19 Protocol Enforcer...")

    # ── Initialize all hardware/modules ───────────────────────────────────
    display         = OLEDDisplay()
    ultrasonic      = UltrasonicSensor(trigger_pin=23, echo_pin=24)
    mask_detector   = MaskDetector(camera_index=0)
    temp_sensor     = TemperatureSensor()
    social_detector = SocialDistancingDetector(camera_index=0)

    display.show_message("CoPE Ready", line2="Monitoring...")
    log.info("All modules initialized. Starting main loop.")

    try:
        while True:
            # ── MODE 1: Idle — show weather, watch for presence ───────────
            log.info("MODE 1: Idle — fetching weather...")
            weather_data = get_weather()
            display_weather_screen(display, weather_data)

            # Poll ultrasonic sensor
            distance = ultrasonic.get_distance_cm()
            log.info(f"Ultrasonic distance: {distance:.1f} cm")

            if distance <= PROXIMITY_THRESHOLD_CM:
                # ── MODE 2: Active — person detected ──────────────────────
                run_active_mode(mask_detector, temp_sensor, social_detector, display)
            else:
                log.info("No presence detected. Staying in idle mode.")

            time.sleep(CYCLE_PAUSE_S)

    except KeyboardInterrupt:
        log.info("CoPE stopped by user.")
    finally:
        ultrasonic.cleanup()
        mask_detector.release()
        social_detector.release()
        display.clear()
        log.info("Cleanup complete. Goodbye.")


if __name__ == "__main__":
    main()
