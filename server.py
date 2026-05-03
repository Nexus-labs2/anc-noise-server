"""
╔══════════════════════════════════════════════════════════════════╗
║   3-MIC NOISE CANCELLATION — SERVER  (Railway.app)              ║
║   Fixed Version                                                   ║
║                                                                   ║
║   Bugs fixed:                                                     ║
║   1. TARGET_SAMPLES was 48000 — DSP never triggered              ║
║      Now uses 512 samples (~32ms chunks)                         ║
║   2. noisereduce blocked event loop — now runs in executor       ║
║   3. WAV base64 payload was ~256KB — too large for Railway       ║
║      Now sends small waveform arrays + WAV only every 3s         ║
║   4. Channel 2 (Mic3) missing from ESP32 but required by         ║
║      server — now works with 2 or 3 mics automatically           ║
╚══════════════════════════════════════════════════════════════════╝
"""

import asyncio
import struct
import json
import os
import io
import wave
import base64
import traceback
from functools import partial
from collections import deque

import numpy as np
import noisereduce as nr
from numpy.fft import rfft
from aiohttp import web, WSMsgType

print("🚀 SERVER STARTED")

# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────
SAMPLE_RATE        = 16000
BUFFER_SIZE        = 512       # samples per DSP chunk (~32ms) — was 48000!
DSP_INTERVAL       = 0.030     # seconds between DSP ticks
MAX_BUFFER         = 16000     # drop oldest samples beyond this (1 sec)
WAV_EVERY_N_CHUNKS = 50        # send WAV audio clip every N chunks (~1.6s)

# ─────────────────────────────────────────────────────────────
#  GLOBAL STATE
# ─────────────────────────────────────────────────────────────
audio_buffers     = {0: deque(), 1: deque(), 2: deque()}
dashboard_clients = set()
esp32_connected   = False
frames_received   = {0: 0, 1: 0, 2: 0}
dsp_chunk_count   = 0
wav_raw_accum     = []
wav_clean_accum   = []
CONTROL           = {"noise_reduction": True}

# ─────────────────────────────────────────────────────────────
#  FRAME PARSER
# ─────────────────────────────────────────────────────────────
def parse_frame(data: bytes):
    if len(data) < 3:
        return None, None, None
    ch    = data[0]
    count = struct.unpack_from('<H', data, 1)[0]
    if ch not in (0, 1, 2):
        return None, None, None
    if len(data) < 3 + count * 2:
        return None, None, None
    try:
        samples = struct.unpack_from(f'<{count}h', data, 3)
        return ch, count, np.array(samples, dtype=np.int16)
    except Exception:
        return None, None, None

# ─────────────────────────────────────────────────────────────
#  DSP HELPERS
# ─────────────────────────────────────────────────────────────
def compute_rms_norm(x: np.ndarray) -> float:
    if len(x) == 0:
        return 0.0
    return float(np.sqrt(np.mean(x.astype(np.float32) ** 2)) / 32768.0)

def compute_fft(samples_f32: np.ndarray, n_bins: int = 80) -> list:
    windowed = samples_f32 * np.hanning(len(samples_f32))
    spectrum = np.abs(rfft(windowed))
    mx = np.max(spectrum) + 1e-9
    return (spectrum[:n_bins] / mx).tolist()

def to_wav_b64(audio_int16: np.ndarray) -> str:
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())
    return base64.b64encode(buf.getvalue()).decode()

def noise_reduce_fn(y: np.ndarray) -> np.ndarray:
    return nr.reduce_noise(y=y, sr=SAMPLE_RATE, stationary=True)

def downsample(arr: np.ndarray, target: int = 200) -> list:
    if len(arr) <= target:
        return arr.tolist()
    step = max(1, len(arr) // target)
    return arr[::step][:target].tolist()

# ─────────────────────────────────────────────────────────────
#  DSP LOOP
# ─────────────────────────────────────────────────────────────
async def process_audio():
    global dsp_chunk_count
    loop = asyncio.get_event_loop()

    while True:
        await asyncio.sleep(DSP_INTERVAL)
        try:
            # Require at least ch0 and ch1 — works with 2 or 3 mics
            if len(audio_buffers[0]) < BUFFER_SIZE:
                continue
            if len(audio_buffers[1]) < BUFFER_SIZE:
                continue

            chunk = {}
            for i in range(3):
                if len(audio_buffers[i]) >= BUFFER_SIZE:
                    raw = [audio_buffers[i].popleft() for _ in range(BUFFER_SIZE)]
                    chunk[i] = np.array(raw, dtype=np.float32) / 32768.0
                else:
                    # Mic3 absent — synthesise from average of Mic1+Mic2
                    chunk[i] = (chunk[0] + chunk[1]) / 2.0

                # Overflow guard
                if len(audio_buffers[i]) > MAX_BUFFER:
                    excess = len(audio_buffers[i]) - MAX_BUFFER
                    for _ in range(excess):
                        audio_buffers[i].popleft()

            # Beamform
            raw_mix = (chunk[0] + chunk[1] + chunk[2]) / 3.0

            # Noise reduction — non-blocking via thread pool
            if CONTROL["noise_reduction"]:
                try:
                    clean_mix = await loop.run_in_executor(
                        None, partial(noise_reduce_fn, raw_mix)
                    )
                except Exception as e:
                    print(f"⚠️  NR failed: {e}")
                    clean_mix = raw_mix
            else:
                clean_mix = raw_mix

            raw_int16   = (raw_mix   * 32767).clip(-32768, 32767).astype(np.int16)
            clean_int16 = (clean_mix * 32767).clip(-32768, 32767).astype(np.int16)

            wav_raw_accum.extend(raw_int16.tolist())
            wav_clean_accum.extend(clean_int16.tolist())

            rms = [
                compute_rms_norm((chunk[i] * 32767).astype(np.int16))
                for i in range(3)
            ]

            fft_raw   = compute_fft(raw_mix)
            fft_clean = compute_fft(clean_mix)

            raw_rms_f   = float(np.sqrt(np.mean(raw_mix   ** 2)) + 1e-9)
            clean_rms_f = float(np.sqrt(np.mean(clean_mix ** 2)))
            nr_pct = round(max(0.0, (1.0 - clean_rms_f / raw_rms_f) * 100), 1)

            dsp_chunk_count += 1

            payload = {
                "rms":              rms,
                "fft_raw":          fft_raw,
                "fft_cleaned":      fft_clean,
                "waveform_raw":     downsample(raw_int16),
                "waveform_cleaned": downsample(clean_int16),
                "noise_reduction":  nr_pct,
                "esp32_connected":  esp32_connected,
                "nr_enabled":       CONTROL["noise_reduction"],
                "chunk":            dsp_chunk_count
            }

            # Attach WAV clip every WAV_EVERY_N_CHUNKS chunks (~1.6s of audio)
            if dsp_chunk_count % WAV_EVERY_N_CHUNKS == 0 and wav_raw_accum:
                payload["audio_raw"]   = to_wav_b64(np.array(wav_raw_accum,   dtype=np.int16))
                payload["audio_clean"] = to_wav_b64(np.array(wav_clean_accum, dtype=np.int16))
                payload["sample_rate"] = SAMPLE_RATE
                wav_raw_accum.clear()
                wav_clean_accum.clear()

            await broadcast_dashboard(payload)

        except Exception as e:
            print(f"❌ DSP ERROR: {e}")
            traceback.print_exc()

# ─────────────────────────────────────────────────────────────
#  BROADCAST
# ─────────────────────────────────────────────────────────────
async def broadcast_dashboard(data: dict):
    if not dashboard_clients:
        return
    msg  = json.dumps(data)
    dead = set()
    for ws in list(dashboard_clients):
        try:
            await ws.send_str(msg)
        except Exception:
            dead.add(ws)
    for ws in dead:
        dashboard_clients.discard(ws)

# ─────────────────────────────────────────────────────────────
#  ESP32 WEBSOCKET  /ws
# ─────────────────────────────────────────────────────────────
async def ws_audio_handler(request):
    global esp32_connected

    if request.headers.get("Upgrade", "").lower() != "websocket":
        return web.Response(text="ESP32 audio WebSocket endpoint", status=200)

    ws = web.WebSocketResponse(autoping=True, max_msg_size=2 ** 20)
    try:
        await ws.prepare(request)
    except Exception as e:
        print(f"❌ ws.prepare failed: {e}")
        return web.Response(status=400)

    esp32_connected = True
    print(f"✅ ESP32 CONNECTED from {request.remote}")

    try:
        await ws.send_str(json.dumps({"status": "connected"}))
    except Exception:
        pass

    try:
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                ch, count, samples = parse_frame(msg.data)
                if ch is not None:
                    audio_buffers[ch].extend(samples.tolist())
                    frames_received[ch] += 1
                    total = sum(frames_received.values())
                    if total <= 9:
                        print(f"📦 ch:{ch} samples:{count} buf0:{len(audio_buffers[0])} buf1:{len(audio_buffers[1])}")
            elif msg.type == WSMsgType.TEXT:
                print(f"📩 ESP32: {msg.data}")
            elif msg.type in (WSMsgType.CLOSED, WSMsgType.ERROR):
                break
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"❌ WS error: {e}")
        traceback.print_exc()
    finally:
        esp32_connected = False
        for i in range(3):
            audio_buffers[i].clear()
        print("🔴 ESP32 DISCONNECTED")

    return ws

# ─────────────────────────────────────────────────────────────
#  DASHBOARD WEBSOCKET  /dashboard
# ─────────────────────────────────────────────────────────────
async def ws_dashboard_handler(request):
    if request.headers.get("Upgrade", "").lower() != "websocket":
        return web.Response(text="Dashboard WebSocket endpoint", status=200)

    ws = web.WebSocketResponse(heartbeat=30, max_msg_size=2 ** 20)
    try:
        await ws.prepare(request)
    except Exception as e:
        print(f"❌ Dashboard prepare failed: {e}")
        return web.Response(status=400)

    dashboard_clients.add(ws)
    print(f"📊 Dashboard connected. Total: {len(dashboard_clients)}")

    try:
        await ws.send_str(json.dumps({
            "esp32_connected": esp32_connected,
            "nr_enabled":      CONTROL["noise_reduction"],
            "status":          "dashboard_ready"
        }))
    except Exception:
        pass

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    if "noise_reduction" in data:
                        CONTROL["noise_reduction"] = bool(data["noise_reduction"])
                        print(f"🎛️  NR: {CONTROL['noise_reduction']}")
                except Exception:
                    pass
            elif msg.type in (WSMsgType.CLOSED, WSMsgType.ERROR):
                break
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"❌ Dashboard error: {e}")
    finally:
        dashboard_clients.discard(ws)
        print(f"📊 Dashboard disconnected. Remaining: {len(dashboard_clients)}")

    return ws

# ─────────────────────────────────────────────────────────────
#  HTTP ROUTES
# ─────────────────────────────────────────────────────────────
async def index(request):
    return web.FileResponse("index.html")

async def health(request):
    return web.Response(
        text=json.dumps({
            "status":          "ok",
            "esp32_connected": esp32_connected,
            "dashboards":      len(dashboard_clients),
            "frames":          dict(frames_received),
            "buffers":         {i: len(audio_buffers[i]) for i in range(3)},
            "dsp_chunks":      dsp_chunk_count,
            "nr_enabled":      CONTROL["noise_reduction"]
        }),
        content_type="application/json"
    )

# ─────────────────────────────────────────────────────────────
#  APP
# ─────────────────────────────────────────────────────────────
app = web.Application(client_max_size=2 * 1024 ** 2)
app.router.add_get('/',          index)
app.router.add_get('/health',    health)
app.router.add_get('/ws',        ws_audio_handler)
app.router.add_get('/dashboard', ws_dashboard_handler)

async def on_startup(app):
    print(f"🎧 DSP engine: BUFFER_SIZE={BUFFER_SIZE} SAMPLE_RATE={SAMPLE_RATE}")
    app["dsp_task"] = asyncio.create_task(process_audio())

async def on_cleanup(app):
    t = app.get("dsp_task")
    if t:
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass

app.on_startup.append(on_startup)
app.on_cleanup.append(on_cleanup)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"🌐 Binding 0.0.0.0:{port}")
    web.run_app(app, host="0.0.0.0", port=port, access_log=None)