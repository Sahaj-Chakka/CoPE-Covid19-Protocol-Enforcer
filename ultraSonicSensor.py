"""
distanceMeasurer_module/ultraSonicSensor.py
============================================
Controls the HC-SR04 Ultrasonic Distance Sensor via Raspberry Pi GPIO.

Wiring:
    HC-SR04 VCC  → Pi 5V  (Pin 2)
    HC-SR04 GND  → Pi GND (Pin 6)
    HC-SR04 TRIG → Pi GPIO 23 (Pin 16)
    HC-SR04 ECHO → Pi GPIO 24 (Pin 18) via voltage divider (5V→3.3V)
"""

import time
import logging

log = logging.getLogger("UltrasonicSensor")

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    log.warning("RPi.GPIO not found — running in simulation mode.")
    GPIO_AVAILABLE = False

SPEED_OF_SOUND_CM_S = 34300  # cm/s at ~20°C


class UltrasonicSensor:
    def __init__(self, trigger_pin: int = 23, echo_pin: int = 24):
        self.trigger_pin = trigger_pin
        self.echo_pin    = echo_pin

        if GPIO_AVAILABLE:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            GPIO.setup(self.trigger_pin, GPIO.OUT)
            GPIO.setup(self.echo_pin,    GPIO.IN)
            GPIO.output(self.trigger_pin, GPIO.LOW)
            time.sleep(0.05)  # Let sensor settle
            log.info(f"Ultrasonic sensor initialized (TRIG={trigger_pin}, ECHO={echo_pin})")

    def get_distance_cm(self) -> float:
        """
        Sends a 10µs pulse and measures echo return time.
        Returns distance in centimeters.
        """
        if not GPIO_AVAILABLE:
            # Simulation: return a fixed distance for testing
            log.debug("Simulation mode: returning 80 cm")
            return 80.0

        # Send 10µs trigger pulse
        GPIO.output(self.trigger_pin, GPIO.HIGH)
        time.sleep(0.00001)
        GPIO.output(self.trigger_pin, GPIO.LOW)

        # Wait for echo to start
        pulse_start = time.time()
        timeout = pulse_start + 0.1
        while GPIO.input(self.echo_pin) == GPIO.LOW:
            pulse_start = time.time()
            if pulse_start > timeout:
                log.warning("Ultrasonic timeout (echo start)")
                return 999.0

        # Wait for echo to end
        pulse_end = time.time()
        timeout = pulse_end + 0.1
        while GPIO.input(self.echo_pin) == GPIO.HIGH:
            pulse_end = time.time()
            if pulse_end > timeout:
                log.warning("Ultrasonic timeout (echo end)")
                return 999.0

        pulse_duration = pulse_end - pulse_start
        distance_cm = (pulse_duration * SPEED_OF_SOUND_CM_S) / 2
        log.debug(f"Distance measured: {distance_cm:.1f} cm")
        return round(distance_cm, 1)

    def is_someone_nearby(self, threshold_cm: float = 150.0) -> bool:
        """Returns True if an object is within threshold distance."""
        return self.get_distance_cm() <= threshold_cm

    def cleanup(self):
        """Release GPIO pins cleanly."""
        if GPIO_AVAILABLE:
            GPIO.cleanup()
            log.info("GPIO cleaned up.")
