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
MAX_BUFFER   = BUFFER_SIZE * 10  # 🔥 prevent infinite buffer growth

audio_buffers = {0: [], 1: [], 2: []}
dashboard_clients = set()

# =========================
# FRAME PARSER
# =========================
def parse_frame(data: bytes):
    if len(data) < 3:
        return None, None, None

    ch = data[0]

    if ch not in (0, 1, 2):  # 🔥 reject invalid channel
        return None, None, None

    count = struct.unpack_from('<H', data, 1)[0]

    if len(data) < 3 + count * 2:  # 🔥 bounds check
        return None, None, None

    try:
        samples = struct.unpack_from(f'<{count}h', data, 3)
    except Exception as e:
        print(f"❌ Frame parse error: {e}")
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
            # 🔥 Trim buffers if they grow too large (stale data)
            for i in range(3):
                if len(audio_buffers[i]) > MAX_BUFFER:
                    audio_buffers[i] = audio_buffers[i][-BUFFER_SIZE:]
                    print(f"⚠️ Buffer {i} overflow — trimmed")

            if not all(len(audio_buffers[i]) >= BUFFER_SIZE for i in range(3)):
                continue  # 🔥 non-blocking skip instead of stalling

            chunk = {}
            for i in range(3):
                data = audio_buffers[i][:BUFFER_SIZE]
                audio_buffers[i] = audio_buffers[i][BUFFER_SIZE:]
                chunk[i] = np.array(data, dtype=np.float32) / 32768.0

            # 🔥 Clamp to [-1, 1] before noise reduction
            for i in range(3):
                chunk[i] = np.clip(chunk[i], -1.0, 1.0)

            cleaned = {}
            for i in range(3):
                try:
                    cleaned[i] = nr.reduce_noise(
                        y=chunk[i],
                        sr=SAMPLE_RATE,
                        stationary=True
                    )
                except Exception as e:
                    print(f"⚠️ Noise reduce failed ch{i}: {e}")
                    cleaned[i] = chunk[i]  # 🔥 fallback to raw

            raw_mix   = (chunk[0]   + chunk[1]   + chunk[2])   / 3.0
            clean_mix = (cleaned[0] + cleaned[1] + cleaned[2]) / 3.0

            raw_int16   = (raw_mix   * 32767).astype(np.int16)
            clean_int16 = (clean_mix * 32767).astype(np.int16)

            rms = [
                compute_rms((chunk[i] * 32767).astype(np.int16))
                for i in range(3)
            ]

            fft_freqs, fft_raw   = compute_fft(raw_mix)
            _,         fft_clean = compute_fft(clean_mix)

            payload = {
                "rms":             rms,
                "fft_freqs":       fft_freqs,
                "fft_raw":         fft_raw,
                "fft_cleaned":     fft_clean,
                "waveform_raw":    raw_int16.tolist(),
                "waveform_cleaned": clean_int16.tolist()
            }

            # 🔥 Non-blocking broadcast
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
    dead = set()

    await asyncio.gather(
        *[ws.send_str(msg) for ws in dashboard_clients],
        return_exceptions=True  # 🔥 one bad client won't crash others
    )

    # Clean dead clients
    for ws in list(dashboard_clients):
        if ws.closed:
            dead.add(ws)

    for ws in dead:
        dashboard_clients.discard(ws)

# =========================
# ESP32 WEBSOCKET HANDLER
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
        async for msg in ws:  # 🔥 cleaner loop, auto handles close/error
            if msg.type == WSMsgType.BINARY:
                ch, count, samples = parse_frame(msg.data)
                if ch is not None:
                    audio_buffers[ch].extend(samples.tolist())

            elif msg.type == WSMsgType.TEXT:
                print("📩 TEXT:", msg.data)

            elif msg.type == WSMsgType.ERROR:
                print("❌ WS ERROR:", ws.exception())
                break

    except Exception as e:
        print("❌ WS EXCEPTION:", e)

    finally:
        # 🔥 Clear buffers on disconnect to avoid stale data
        for i in range(3):
            audio_buffers[i].clear()
        print("🔴 ESP32 DISCONNECTED — buffers cleared")

    return ws

# =========================
# DASHBOARD WS HANDLER
# =========================
async def ws_dashboard_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    dashboard_clients.add(ws)
    print("📊 Dashboard connected — total:", len(dashboard_clients))

    try:
        async for _ in ws:
            pass
    except Exception:
        pass
    finally:
        dashboard_clients.discard(ws)
        print("📊 Dashboard disconnected — total:", len(dashboard_clients))

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

app.router.add_get('/',          index)
app.router.add_get('/health',    health)
app.router.add_get('/ws',        ws_audio_handler)
app.router.add_get('/dashboard', ws_dashboard_handler)

# =========================
# STARTUP / CLEANUP
# =========================
async def delayed_start(app):
    print("🚀 Server started...")
    await asyncio.sleep(2)
    print("🎧 DSP engine starting...")
    app["task"] = asyncio.create_task(process_audio())

async def cleanup(app):
    if "task" in app:
        app["task"].cancel()
        try:
            await app["task"]
        except asyncio.CancelledError:
            pass
        print("🛑 DSP task cancelled cleanly")

app.on_startup.append(delayed_start)
app.on_cleanup.append(cleanup)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print("🌐 Listening on port", port)
    web.run_app(app, host="0.0.0.0", port=port)