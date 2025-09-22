// MineSafe AI Assistant - JavaScript Backend

// ---------------- CONFIG ----------------
const EARTHQUAKE_API_KEY = "fb10922e-9fa9-4bfe-a44a-7aecc17d5234";
const WEATHER_API_KEY = "620a81f2719043c0bfe180055250609";
const GEMINI_API_KEY = "AIzaSyAAWVACXCCvMBuLpjqyLGgx0hwhyibDpNU"; // Replace with your actual Gemini API key

// Full knowledge base
const KNOWLEDGE_BASE = `... your mining safety text here ...`;

// ---------------- UTILITY FUNCTIONS ----------------
function toRadians(deg) { return deg * (Math.PI/180); }
function haversine(lat1, lon1, lat2, lon2) {
  const R = 6371;
  const dLat = toRadians(lat2 - lat1);
  const dLon = toRadians(lon2 - lon1);
  const a = Math.sin(dLat/2)**2 +
            Math.cos(toRadians(lat1)) * Math.cos(toRadians(lat2)) *
            Math.sin(dLon/2)**2;
  return 2 * R * Math.asin(Math.sqrt(a));
}

// Get coordinates from PIN code
async function getCoordinatesFromPincode(pin) {
  const res = await fetch(`https://nominatim.openstreetmap.org/search?postalcode=${pin}&countrycodes=in&format=json`, {
    headers: { "User-Agent": "MineSafe-App/1.0" }
  });
  const data = await res.json();
  if (data.length === 0) throw new Error("PIN not found");
  return { lat: parseFloat(data[0].lat), lon: parseFloat(data[0].lon) };
}

// Earthquake check
async function checkEarthquakes(userLat, userLon, radiusKm=50) {
  try {
    const res = await fetch("https://api.apiverve.com/v1/earthquake", {
      headers: { "X-API-Key": EARTHQUAKE_API_KEY }
    });
    const data = await res.json();
    if (data.status !== "ok") return false;
    for (const quake of data.data?.earthquakes || []) {
      const [lon, lat] = quake.coordinates;
      if (haversine(userLat, userLon, lat, lon) <= radiusKm) return true;
    }
    return false;
  } catch { return false; }
}

// Weather check
async function getWeather(lat, lon) {
  const res = await fetch(`http://api.weatherapi.com/v1/current.json?key=${WEATHER_API_KEY}&q=${lat},${lon}&aqi=no`);
  const data = await res.json();
  if (data.error) throw new Error(data.error.message);
  return { temp_c: data.current.temp_c, rain_mm: data.current.precip_mm || 0 };
}

// ---------------- GEMINI ----------------
async function callGeminiText(prompt) {
  try {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${GEMINI_API_KEY}`;
    const body = {
      contents: [
        { role: "user", parts: [{ text: `Knowledge Base:\n${KNOWLEDGE_BASE}\n\nUser: ${prompt}` }] }
      ]
    };
    const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const data = await res.json();
    if (data.error) throw new Error(data.error.message);
    return data.candidates[0].content.parts[0].text.trim();
  } catch (err) {
    console.error("Gemini error:", err);
    return `❌ Gemini API error: ${err.message}`;
  }
}

// ---------------- CORE FUNCTIONS ----------------
async function performSafetyCheck(pin) {
  try {
    const { lat, lon } = await getCoordinatesFromPincode(pin);
    const reasons = [];

    if (await checkEarthquakes(lat, lon)) reasons.push("- Earthquake within 50 km");

    const { temp_c, rain_mm } = await getWeather(lat, lon);
    if (temp_c <= 0) reasons.push(`- Too cold (${temp_c}°C)`);
    else if (temp_c >= 40) reasons.push(`- Too hot (${temp_c}°C)`);
    if (rain_mm > 0) reasons.push(`- Rainfall ${rain_mm} mm`);

    if (reasons.length > 0) {
      return { safe:false, message:`❌ Not safe today:\n${reasons.join("\n")}` };
    }
    return { safe:true, message:`✅ Safe today. Temp: ${temp_c}°C. No rain. No quakes nearby.` };
  } catch (err) {
    return { safe:false, message:`❌ Error: ${err.message}` };
  }
}

// ---------------- QUERY ROUTER ----------------
function answerQuery(q) {
  const lq = q.toLowerCase().trim();

  // Greeting handling
  if (["hi", "hello", "hey"].includes(lq)) {
    return { type: "info", message: 
      "Hello! I'm here to provide information and guidance on mining safety. How can I assist you today?" 
    };
  }

  if (lq.includes("safe to work")) 
    return { type:"safety_check", message:"Please enter your 6-digit PIN." };

  if (lq.includes("risk")) 
    return { type:"info", message:"Current risk: MEDIUM. Enhanced monitoring in effect." };

  if (lq.includes("alert")) 
    return { type:"info", message:"Alerts:\n• Weather advisory\n• Seismic low activity 75km away\n• Equipment maintenance pending" };

  if (lq.includes("csv") || lq.includes("upload")) 
    return { type:"info", message:"CSV must include: timestamp, sensor_id, temp, humidity, pressure, location_id." };

  // Fallback → Gemini
  return { type:"gemini" };
}

// ---------------- HANDLERS ----------------
async function handleChatMessage(msg) {
  const resp = answerQuery(msg);
  if (resp.type === "safety_check") return { needsPincode:true, message:resp.message };
  if (resp.type === "gemini") return { needsPincode:false, message:await callGeminiText(msg) };
  return { needsPincode:false, message:resp.message };
}

async function handleSafetyCheckWithPincode(pin) { 
  return { needsPincode:false, message:(await performSafetyCheck(pin)).message }; 
}

// Export
if (typeof window !== "undefined") {
  window.MineSafeAI = { handleChatMessage, handleSafetyCheckWithPincode };
}
