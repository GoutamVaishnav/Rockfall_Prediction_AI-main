import requests
import math

# ---------------- CONFIG ----------------
EARTHQUAKE_API_KEY = "fb10922e-9fa9-4bfe-a44a-7aecc17d5234"
WEATHER_API_KEY = "620a81f2719043c0bfe180055250609"

# ---------------- FUNCTIONS ----------------
def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance between two lat/lon points in km"""
    R = 6371  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) *
         math.sin(dlon/2)**2)
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def get_coordinates_from_pincode(pincode):
    """Get lat/lon from PIN code using Nominatim"""
    geocode_url = f"https://nominatim.openstreetmap.org/search?postalcode={pincode}&countrycodes=in&format=json"
    geo_resp = requests.get(geocode_url, headers={"User-Agent": "MyApp"}).json()
    if geo_resp:
        return float(geo_resp[0]["lat"]), float(geo_resp[0]["lon"])
    else:
        raise ValueError("Could not find coordinates for that PIN code")

def check_earthquakes(user_lat, user_lon, radius_km=50):
    """Check earthquakes within given radius (km)"""
    eq_url = "https://api.apiverve.com/v1/earthquake"
    headers = {"X-API-Key": EARTHQUAKE_API_KEY}
    eq_resp = requests.get(eq_url, headers=headers).json()
    nearby_quake = False
    if eq_resp["status"] == "ok":
        earthquakes = eq_resp["data"]["earthquakes"]
        for quake in earthquakes:
            lon, lat = quake["coordinates"]
            distance = haversine(user_lat, user_lon, lat, lon)
            if distance <= radius_km:
                nearby_quake = True
                break
    return nearby_quake

def get_weather(user_lat, user_lon):
    """Get current temperature and rainfall from WeatherAPI"""
    url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={user_lat},{user_lon}&aqi=no"
    resp = requests.get(url).json()
    
    # Check if API returned an error
    if "error" in resp:
        msg = resp["error"].get("message", "Unknown error from WeatherAPI")
        raise ValueError(f"WeatherAPI error: {msg}")
    
    temp_c = resp["current"]["temp_c"]
    rain_mm = resp["current"].get("precip_mm", 0)
    return temp_c, rain_mm


# ---------------- MAIN ----------------
def main():
    pincode = input("Enter your PIN code: ")
    try:
        user_lat, user_lon = get_coordinates_from_pincode(pincode)
    except ValueError as e:
        print("❌", e)
        return

    # Check each safety condition
    reasons = []

    # Earthquake
    if check_earthquakes(user_lat, user_lon):
        reasons.append("- Earthquake detected within 50 km")

    # Weather
    temp_c, rain_mm = get_weather(user_lat, user_lon)
    if temp_c <= 0:
        reasons.append(f"- Temperature too low: {temp_c}°C")
    elif temp_c >= 40:
        reasons.append(f"- Temperature too high: {temp_c}°C")
    if rain_mm > 0:
        reasons.append(f"- Rainfall: {rain_mm} mm")

    # Final output
    if reasons:
        print("\n❌ Not safe to work today.\nReasons:")
        for r in reasons:
            print(r)
    else:
        print(f"\n✅ Safe to work today.\nNo rainfall at this moment.\nTemperature within the safe range: {temp_c}°C.\nNo earthquake within 50KM range.")

# ---------------- RUN ----------------
if __name__ == "__main__":
    main()

