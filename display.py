"""
display_module/display.py
==========================
Controls the SSD1306 mini OLED display via I2C on Raspberry Pi.

Wiring (shares I2C bus with IR temp sensor):
    OLED VCC → 3.3V
    OLED GND → GND
    OLED SDA → Pi SDA (GPIO 2, Pin 3)
    OLED SCL → Pi SCL (GPIO 3, Pin 5)

Install: pip install adafruit-circuitpython-ssd1306 pillow
"""

import logging
import time

log = logging.getLogger("OLEDDisplay")

OLED_WIDTH  = 128
OLED_HEIGHT = 64

try:
    import board
    import busio
    import adafruit_ssd1306
    from PIL import Image, ImageDraw, ImageFont
    OLED_AVAILABLE = True
except ImportError:
    log.warning("OLED libraries not found — display running in simulation mode.")
    OLED_AVAILABLE = False


class OLEDDisplay:
    def __init__(self, width: int = OLED_WIDTH, height: int = OLED_HEIGHT):
        self.width  = width
        self.height = height
        self.oled   = None

        if OLED_AVAILABLE:
            try:
                i2c       = busio.I2C(board.SCL, board.SDA)
                self.oled = adafruit_ssd1306.SSD1306_I2C(width, height, i2c)
                self.oled.fill(0)
                self.oled.show()
                log.info("OLED display initialized.")
            except Exception as e:
                log.error(f"OLED init failed: {e}")
                self.oled = None

        # Create image buffer
        self.image = Image.new("1", (width, height))
        self.draw  = ImageDraw.Draw(self.image)

        # Load font (fallback to default if not found)
        try:
            self.font       = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
            self.font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except Exception:
            self.font       = ImageFont.load_default()
            self.font_large = self.font

    def _refresh(self):
        """Push image buffer to the physical OLED."""
        if self.oled:
            self.oled.image(self.image)
            self.oled.show()
        else:
            # Simulation: print to console
            log.info("[OLED] Display updated (simulation mode)")

    def clear(self):
        """Clear the display."""
        self.draw.rectangle((0, 0, self.width, self.height), outline=0, fill=0)
        self._refresh()

    def show_message(self, line1: str, line2: str = "", line3: str = ""):
        """
        Display up to 3 lines of text on the OLED.

        Args:
            line1: Primary message (displayed larger)
            line2: Secondary message
            line3: Optional third line
        """
        self.clear()
        self.draw.text((0, 0),  line1, font=self.font_large, fill=255)
        self.draw.text((0, 20), line2, font=self.font,       fill=255)
        self.draw.text((0, 40), line3, font=self.font,       fill=255)
        self._refresh()
        log.debug(f"OLED: '{line1}' | '{line2}' | '{line3}'")

    def show_weather(self, temp: float, condition: str, time_str: str = ""):
        """
        Display weather info in idle mode (Mode 1).
        Layout:
            [time_str]
            [temp]°C
            [condition]
        """
        self.clear()
        self.draw.text((0, 0),  time_str,              font=self.font,       fill=255)
        self.draw.text((0, 15), f"{temp:.0f}°C",       font=self.font_large, fill=255)
        self.draw.text((0, 38), condition[:20],         font=self.font,       fill=255)
        self._refresh()

    def show_temperature_result(self, temp_c: float, is_fever: bool):
        """Dedicated temperature result display."""
        self.clear()
        status = "⚠ HIGH TEMP" if is_fever else "Normal ✓"
        self.draw.text((0, 0),  f"Temp: {temp_c:.1f}C", font=self.font_large, fill=255)
        self.draw.text((0, 25), status,                  font=self.font,       fill=255)
        self._refresh()

    def scroll_text(self, text: str, delay: float = 0.05):
        """
        Scroll long text across the display horizontally.
        Useful for displaying longer status messages.
        """
        padded = " " * 20 + text + " " * 20
        for i in range(len(padded) - 20):
            self.clear()
            self.draw.text((0, 25), padded[i:i+20], font=self.font, fill=255)
            self._refresh()
            time.sleep(delay)
