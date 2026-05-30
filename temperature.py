"""
temperature_module/temperature.py
===================================
Reads body temperature using an IR (Infrared) sensor via I2C on Raspberry Pi.

Wiring (I2C shared with OLED on SDA/SCL pins):
    Sensor VCC → 3.3V (Pin 1)
    Sensor GND → GND  (Pin 6)
    Sensor SDA → Pi SDA (Pin 3, GPIO 2)
    Sensor SCL → Pi SCL (Pin 5, GPIO 3)

Common IR temp sensor: MLX90614 (I2C address: 0x5A)
Enable I2C: sudo raspi-config → Interface Options → I2C → Enable
"""

import time
import logging

log = logging.getLogger("TemperatureSensor")

# MLX90614 I2C register addresses
MLX90614_ADDR    = 0x5A
MLX90614_TOBJ1   = 0x07   # Object (body) temperature register
MLX90614_TAMB    = 0x06   # Ambient temperature register

try:
    import smbus2
    SMBUS_AVAILABLE = True
except ImportError:
    log.warning("smbus2 not found — running in simulation mode.")
    SMBUS_AVAILABLE = False


class TemperatureSensor:
    def __init__(self, i2c_bus: int = 1, address: int = MLX90614_ADDR):
        self.address = address

        if SMBUS_AVAILABLE:
            try:
                self.bus = smbus2.SMBus(i2c_bus)
                log.info(f"IR Temperature sensor initialized (I2C bus {i2c_bus}, addr 0x{address:02X})")
            except Exception as e:
                log.error(f"Failed to open I2C bus: {e}")
                self.bus = None
        else:
            self.bus = None

    def _read_raw(self, register: int) -> int:
        """Read 2 bytes from the sensor register and return raw value."""
        data = self.bus.read_i2c_block_data(self.address, register, 3)
        # MLX90614 returns 3 bytes: [LSB, MSB, PEC]
        raw = (data[1] << 8) | data[0]
        return raw

    def _raw_to_celsius(self, raw: int) -> float:
        """Convert raw sensor value to Celsius."""
        # MLX90614 formula: temp_K = raw * 0.02, then subtract 273.15
        temp_k = raw * 0.02
        return round(temp_k - 273.15, 2)

    def read_object_temperature(self) -> float:
        """
        Read a single object (body) temperature reading in Celsius.
        """
        if self.bus is None:
            # Simulation: return a realistic body temp
            import random
            return round(36.5 + random.uniform(-0.5, 1.5), 1)

        try:
            raw  = self._read_raw(MLX90614_TOBJ1)
            temp = self._raw_to_celsius(raw)
            log.debug(f"Single temperature reading: {temp:.1f}°C")
            return temp
        except Exception as e:
            log.error(f"Temperature read error: {e}")
            return -1.0

    def read_ambient_temperature(self) -> float:
        """Read the ambient (room) temperature in Celsius."""
        if self.bus is None:
            return 22.0
        try:
            raw  = self._read_raw(MLX90614_TAMB)
            temp = self._raw_to_celsius(raw)
            log.debug(f"Ambient temperature: {temp:.1f}°C")
            return temp
        except Exception as e:
            log.error(f"Ambient temp read error: {e}")
            return -1.0

    def read_average_temperature(self, readings: int = 5, delay: float = 0.3) -> float:
        """
        Takes multiple readings and returns the average.
        Filters outliers (readings more than 2°C from median).

        Args:
            readings : Number of readings to average (default: 5 as per project spec)
            delay    : Delay between readings in seconds

        Returns:
            Average temperature in Celsius
        """
        log.info(f"Taking {readings} temperature readings...")
        samples = []

        for i in range(readings):
            temp = self.read_object_temperature()
            if temp > 0:
                samples.append(temp)
                log.debug(f"  Reading {i+1}/{readings}: {temp:.1f}°C")
            time.sleep(delay)

        if not samples:
            log.error("No valid temperature readings obtained.")
            return -1.0

        # Filter outliers: keep readings within 2°C of median
        median    = sorted(samples)[len(samples) // 2]
        filtered  = [s for s in samples if abs(s - median) <= 2.0]
        average   = sum(filtered) / len(filtered)

        log.info(f"Average temperature (from {len(filtered)} valid readings): {average:.1f}°C")
        return round(average, 1)

    def is_fever(self, temp_c: float, threshold: float = 37.5) -> bool:
        """Returns True if temperature exceeds fever threshold."""
        return temp_c > threshold
