"""
weatherService_module/getWeather.py
=====================================
Fetches current weather from OpenWeatherMap API.
Displayed on OLED in idle mode (Mode 1).

Get a free API key at: https://openweathermap.org/api
Set your key and city in environment variables.
"""

import os
import logging
import requests

log = logging.getLogger("WeatherService")

API_KEY  = os.environ.get("OPENWEATHER_API_KEY", "your_api_key_here")
CITY     = os.environ.get("COPE_CITY", "Agartala")
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"


def get_weather() -> dict:
    """
    Fetch current weather for the configured city.

    Returns:
        dict with keys: temp_c, condition, humidity, city
        Returns fallback data if API call fails.
    """
    if API_KEY == "your_api_key_here":
        log.warning("OpenWeatherMap API key not set. Using fallback data.")
        return _fallback_weather()

    try:
        params   = {"q": CITY, "appid": API_KEY, "units": "metric"}
        response = requests.get(BASE_URL, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()

        weather = {
            "temp_c":    round(data["main"]["temp"], 1),
            "condition": data["weather"][0]["main"],
            "humidity":  data["main"]["humidity"],
            "city":      data["name"]
        }
        log.info(f"Weather fetched: {weather['temp_c']}°C, {weather['condition']}")
        return weather

    except requests.exceptions.Timeout:
        log.warning("Weather API timeout — using fallback.")
    except requests.exceptions.ConnectionError:
        log.warning("No internet connection — using fallback.")
    except Exception as e:
        log.error(f"Weather API error: {e}")

    return _fallback_weather()


def _fallback_weather() -> dict:
    """Return placeholder weather when API is unavailable."""
    return {
        "temp_c":    "--",
        "condition": "Unavailable",
        "humidity":  "--",
        "city":      CITY
    }
