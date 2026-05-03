import asyncio
import struct
import json
import os
import numpy as np
import noisereduce as nr
from numpy.fft import rfft   # ✅ REPLACED SCIPY
from aiohttp import web, WSMsgType
import io
import wave
import base64

print("🚀 SERVER STARTED")  # ✅ DEBUG START

# =========================
# CONFIG
# =========================
SAMPLE_RATE = 16000
BUFFER_SECONDS = 3
TARGET_SAMPLES = SAMPLE_RATE * BUFFER_SECONDS

audio_buffers = {0: [], 1: [], 2: []}
dashboard_clients = set()

CONTROL = {
    "noise_reduction": True
}

# =========================
# FRAME PARSER
# =========================
def parse_frame(data: bytes):
    if len(data) < 3:
        return None, None, None

    ch = data[0]
    count = struct.unpack_from('<H', data, 1)[0]

    try:
        samples = struct.unpack_from(f'<{count}h', data, 3)
        return ch, count, np.array(samples, dtype=np.int16)
    except:
        return None, None, None

# =========================
# WAV CONVERSION
# =========================
def to_wav(audio):
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())
    return buffer.getvalue()

# =========================
# RMS (NORMALIZED)
# =========================
def compute_rms(x):
    rms = np.sqrt(np.mean(x.astype(np.float32)**2))
    return float(rms / 32768.0)

# =========================
# FFT (NUMPY ONLY)
# =========================
def compute_fft(samples):
    spectrum = np.abs(rfft(samples))
    spectrum = spectrum / (np.max(spectrum) + 1e-6)
    return spectrum[:100].tolist()

# =========================
# DSP LOOP
# =========================
async def process_audio():
    while True:
        await asyncio.sleep(0.1)

        try:
            if not all(len(audio_buffers[i]) >= TARGET_SAMPLES for i in range(3)):
                continue

            chunk = {}
            for i in range(3):
                data = audio_buffers[i][:TARGET_SAMPLES]
                audio_buffers[i] = audio_buffers[i][TARGET_SAMPLES:]
                chunk[i] = np.array(data, dtype=np.float32) / 32768.0

            # Mix
            raw_mix = (chunk[0] + chunk[1] + chunk[2]) / 3.0

            # ✅ SAFE NOISE REDUCTION
            if CONTROL["noise_reduction"]:
                try:
                    clean_mix = nr.reduce_noise(
                        y=raw_mix,
                        sr=SAMPLE_RATE
                    )
                except Exception as e:
                    print("⚠️ NR failed:", e)
                    clean_mix = raw_mix
            else:
                clean_mix = raw_mix

            # Convert back
            raw_int16 = (raw_mix * 32767).astype(np.int16)
            clean_int16 = (clean_mix * 32767).astype(np.int16)

            # WAV encode
            raw_wav = to_wav(raw_int16)
            clean_wav = to_wav(clean_int16)

            raw_b64 = base64.b64encode(raw_wav).decode()
            clean_b64 = base64.b64encode(clean_wav).decode()

            # RMS
            rms = [
                compute_rms((chunk[i] * 32767).astype(np.int16))
                for i in range(3)
            ]

            # FFT
            fft_raw = compute_fft(raw_mix)
            fft_clean = compute_fft(clean_mix)

            payload = {
                "rms": rms,
                "fft_raw": fft_raw,
                "fft_cleaned": fft_clean,
                "audio_raw": raw_b64,
                "audio_clean": clean_b64,
                "sample_rate": SAMPLE_RATE
            }

            asyncio.create_task(broadcast_dashboard(payload))

        except Exception as e:
            print("❌ DSP ERROR:", e)

# =========================
# BROADCAST
# =========================
async def broadcast_dashboard(data):
    if not dashboard_clients:
        return

    msg = json.dumps(data)

    await asyncio.gather(
        *[ws.send_str(msg) for ws in dashboard_clients],
        return_exceptions=True
    )

# =========================
# ESP32 WS
# =========================
async def ws_audio_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    print("✅ ESP32 CONNECTED")

    try:
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                ch, count, samples = parse_frame(msg.data)
                if ch is not None:
                    audio_buffers[ch].extend(samples.tolist())

    finally:
        for i in range(3):
            audio_buffers[i].clear()
        print("🔴 ESP32 DISCONNECTED")

    return ws

# =========================
# DASHBOARD WS
# =========================
async def ws_dashboard_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    dashboard_clients.add(ws)

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                data = json.loads(msg.data)

                if "noise_reduction" in data:
                    CONTROL["noise_reduction"] = data["noise_reduction"]
                    print("🎛️ NR:", CONTROL["noise_reduction"])

    finally:
        dashboard_clients.discard(ws)

    return ws

# =========================
# ROUTES
# =========================
async def index(request):
    return web.FileResponse("index.html")

# ✅ HEALTHCHECK FIX
async def health(request):
    return web.Response(text="OK")

# =========================
# APP INIT
# =========================
app = web.Application()

app.router.add_get('/', index)
app.router.add_get('/health', health)   # ✅ IMPORTANT
app.router.add_get('/ws', ws_audio_handler)
app.router.add_get('/dashboard', ws_dashboard_handler)

# =========================
# START BACKGROUND TASK
# =========================
async def start_bg(app):
    app["task"] = asyncio.create_task(process_audio())

app.on_startup.append(start_bg)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print("🌐 Listening on port", port)
    web.run_app(app, host="0.0.0.0", port=port)