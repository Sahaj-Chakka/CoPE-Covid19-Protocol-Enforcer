"""
displayWeather.py
==================
Displays weather information on the OLED in idle Mode 1.
Called from startExecution.py when no one is nearby.
"""

import logging
from datetime import datetime

log = logging.getLogger("DisplayWeather")


def display_weather_screen(display, weather_data: dict):
    """
    Show current weather on the OLED display.

    Args:
        display      : OLEDDisplay instance
        weather_data : dict from get_weather()
    """
    time_str  = datetime.now().strftime("%H:%M")
    temp      = weather_data.get("temp_c", "--")
    condition = weather_data.get("condition", "N/A")

    log.info(f"Showing weather: {temp}°C, {condition} at {time_str}")
    display.show_weather(temp, condition, time_str)
