import asyncio
import struct
import json
import os
import numpy as np
import noisereduce as nr
from scipy.fft import rfft, rfftfreq
from aiohttp import web, WSMsgType

# =========================
# GLOBAL CONFIG
# =========================
SAMPLE_RATE = 16000
BUFFER_SIZE = 1024

audio_buffers = {0: [], 1: [], 2: []}
dashboard_clients = set()

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
    except:
        return None, None, None

    return ch, count, np.array(samples, dtype=np.int16)

# =========================
# RMS
# =========================
def compute_rms(x):
    return int(np.sqrt(np.mean(x.astype(np.float32) ** 2)))

# =========================
# FFT
# =========================
def compute_fft(samples):
    windowed = samples * np.hanning(len(samples))
    spectrum = np.abs(rfft(windowed))
    freqs = rfftfreq(len(samples), 1 / SAMPLE_RATE)

    half = len(freqs) // 2
    return freqs[:half].tolist(), spectrum[:half].tolist()

# =========================
# DSP LOOP
# =========================
async def process_audio():
    while True:
        await asyncio.sleep(0.03)

        try:
            if all(len(audio_buffers[i]) >= BUFFER_SIZE for i in range(3)):

                chunk = {}
                for i in range(3):
                    data = audio_buffers[i][:BUFFER_SIZE]
                    audio_buffers[i] = audio_buffers[i][BUFFER_SIZE:]
                    chunk[i] = np.array(data, dtype=np.float32) / 32768.0

                cleaned = {}
                for i in range(3):
                    cleaned[i] = nr.reduce_noise(
                        y=chunk[i],
                        sr=SAMPLE_RATE,
                        stationary=True
                    )

                raw_mix = (chunk[0] + chunk[1] + chunk[2]) / 3
                clean_mix = (cleaned[0] + cleaned[1] + cleaned[2]) / 3

                raw_int16 = (raw_mix * 32767).astype(np.int16)
                clean_int16 = (clean_mix * 32767).astype(np.int16)

                rms = [
                    compute_rms((chunk[i] * 32767).astype(np.int16))
                    for i in range(3)
                ]

                fft_freqs, fft_raw = compute_fft(raw_mix)
                _, fft_clean = compute_fft(clean_mix)

                payload = {
                    "rms": rms,
                    "fft_freqs": fft_freqs,
                    "fft_raw": fft_raw,
                    "fft_cleaned": fft_clean,
                    "waveform_raw": raw_int16.tolist(),
                    "waveform_cleaned": clean_int16.tolist()
                }

                await broadcast_dashboard(payload)

        except Exception as e:
            print("❌ DSP ERROR:", e)

# =========================
# BROADCAST
# =========================
async def broadcast_dashboard(data):
    dead = set()

    for ws in dashboard_clients:
        try:
            await ws.send_str(json.dumps(data))
        except:
            dead.add(ws)

    for ws in dead:
        dashboard_clients.discard(ws)

# =========================
# ESP32 WEBSOCKET (FINAL FIX)
# =========================
async def ws_audio_handler(request):

    print("🔌 Incoming WS request from:", request.remote)

    ws = web.WebSocketResponse(
        autoping=True,
        heartbeat=20,
        max_msg_size=2**20
    )

    await ws.prepare(request)

    print("✅ ESP32 CONNECTED")

    try:
        while True:
            msg = await ws.receive()

            if msg.type == WSMsgType.BINARY:
                ch, count, samples = parse_frame(msg.data)

                if ch is not None:
                    audio_buffers[ch].extend(samples.tolist())

            elif msg.type == WSMsgType.TEXT:
                print("📩 TEXT:", msg.data)

            elif msg.type == WSMsgType.CLOSED:
                print("⚠️ WS CLOSED")
                break

            elif msg.type == WSMsgType.ERROR:
                print("❌ WS ERROR:", ws.exception())
                break

    except Exception as e:
        print("❌ WS EXCEPTION:", e)

    print("🔴 ESP32 DISCONNECTED")
    return ws

# =========================
# DASHBOARD WS
# =========================
async def ws_dashboard_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    dashboard_clients.add(ws)
    print("📊 Dashboard connected")

    try:
        async for _ in ws:
            pass
    except:
        pass

    dashboard_clients.discard(ws)
    print("📊 Dashboard disconnected")

    return ws

# =========================
# ROUTES
# =========================
async def index(request):
    return web.Response(text="Server Running", status=200)

async def health(request):
    return web.Response(text="OK", status=200)

# =========================
# APP INIT
# =========================
app = web.Application(client_max_size=1024**2)

app.router.add_get('/', index)
app.router.add_get('/health', health)
app.router.add_get('/ws', ws_audio_handler)
app.router.add_get('/dashboard', ws_dashboard_handler)

# =========================
# STARTUP
# =========================
async def delayed_start(app):
    print("🚀 Server started...")
    await asyncio.sleep(2)
    print("🎧 DSP engine starting...")
    app["task"] = asyncio.create_task(process_audio())

async def cleanup(app):
    if "task" in app:
        app["task"].cancel()

app.on_startup.append(delayed_start)
app.on_cleanup.append(cleanup)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print("🌐 Listening on port", port)
    web.run_app(app, host="0.0.0.0", port=port)