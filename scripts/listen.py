"""
The listening tester (roadmap idea 58).

pluginval and the CLAP validator check that a plug-in loads and answers its API.
Neither listens to it. This does: it loads the released VST3 headless, plays it
test signals the way a host would, writes what came out as WAV files, and looks
at the audio for the faults a producer would hear.

Hard failures, which hold a release back:
  - NaN or infinite samples (that is a crash waiting in someone's session)
  - a silent output when a clear signal went in (the plug-in eats the audio)

Warnings, reported but never blocking, because a creative effect can do these on
purpose and a false alarm must not stop a release:
  - clicks: a sample-to-sample jump far out of line with the rest of a smooth sine
  - sound from silence (self-oscillation, noise, a stuck tail)
  - DC offset
  - sustained clipping at full scale

Usage: python listen.py <path-to.vst3> <results-dir>
"""
import json, math, sys, pathlib
import numpy as np
import soundfile as sf
from pedalboard import load_plugin
from mido import Message

SR = 48000
SECONDS = 4.0
BLOCK = 512

vst3, out_dir = sys.argv[1], pathlib.Path(sys.argv[2])
audio_dir = out_dir / "audio"
audio_dir.mkdir(parents=True, exist_ok=True)


def db(x):
    return -200.0 if x <= 1e-10 else 20 * math.log10(x)


def signals():
    n = int(SR * SECONDS)
    t = np.arange(n) / SR
    rng = np.random.default_rng(7)
    sine = 0.25 * np.sin(2 * np.pi * 1000 * t)                     # -12 dBFS, smooth: any click shows
    white = rng.standard_normal(n)
    # pink-ish noise by a simple one-pole cascade, at about -18 dBFS
    pink = np.zeros(n); b = 0.0
    for i in range(n):
        b = 0.997 * b + white[i] * 0.05
        pink[i] = b
    pink = pink / (np.max(np.abs(pink)) + 1e-9) * 0.125
    # a hardstyle-ish loop at 150 BPM: a pitched-down kick every beat and a hat off-beat
    loop = np.zeros(n)
    beat = int(SR * 60 / 150)
    for start in range(0, n, beat):
        k = np.arange(min(int(SR * 0.25), n - start))
        f = 55 + 300 * np.exp(-k / (SR * 0.02))
        loop[start:start + len(k)] += 0.7 * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-k / (SR * 0.12))
        h = start + beat // 2
        if h < n:
            m = np.arange(min(int(SR * 0.03), n - h))
            loop[h:h + len(m)] += 0.15 * rng.standard_normal(len(m)) * np.exp(-m / (SR * 0.008))
    silence = np.zeros(n)
    stereo = lambda x: np.stack([x, x]).astype(np.float32)
    return {"silence": stereo(silence), "sine_1k": stereo(sine), "pink_noise": stereo(pink), "kick_loop": stereo(loop)}


def analyse(name, x_in, y):
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(y)
    nonfinite = int((~finite).sum())
    y = np.where(finite, y, 0.0)
    skip = int(SR * 0.1)                                           # ignore the first 100 ms of settling
    body = y[:, skip:]
    peak, rms = float(np.max(np.abs(body))), float(np.sqrt(np.mean(body ** 2)))
    in_rms = float(np.sqrt(np.mean(np.asarray(x_in, dtype=np.float64) ** 2))) if x_in is not None else None
    dc = float(np.max(np.abs(np.mean(body, axis=1))))
    clip_frac = float(np.mean(np.abs(body) >= 0.999))

    clicks = 0
    if name == "sine_1k":
        d2 = np.abs(np.diff(body, n=2, axis=1))
        typical = np.median(d2, axis=1, keepdims=True) + 1e-9
        clicks = int((d2 > typical * 60).sum())

    hard, warn = [], []
    if nonfinite:
        hard.append(f"{nonfinite} NaN or infinite samples")
    if in_rms is not None and in_rms > 0.01 and rms < 10 ** (-90 / 20):
        hard.append(f"silent output ({db(rms):.0f} dBFS) from a clear input ({db(in_rms):.0f} dBFS)")
    if name == "silence" and rms > 10 ** (-80 / 20):
        warn.append(f"makes sound from silence ({db(rms):.0f} dBFS RMS)")
    if clicks:
        warn.append(f"{clicks} click(s) in a clean 1 kHz sine")
    if dc > 0.01:
        warn.append(f"DC offset {dc:.3f}")
    if clip_frac > 0.001:
        warn.append(f"{clip_frac * 100:.2f}% of samples at full scale")

    sf.write(audio_dir / f"{name}.wav", y.T.astype(np.float32), SR, subtype="FLOAT")
    return {"signal": name, "peak_dbfs": round(db(peak), 1), "rms_dbfs": round(db(rms), 1),
            "nonfinite": nonfinite, "clicks": clicks, "dc": round(dc, 4), "clip_pct": round(clip_frac * 100, 3),
            "hard": hard, "warn": warn}


result = {"plugin": vst3, "sample_rate": SR, "tests": [], "hard": [], "warn": [], "error": None}
try:
    plugin = load_plugin(vst3)
    result["instrument"] = bool(plugin.is_instrument)
    if plugin.is_instrument:
        # an instrument gets notes, not audio: one C1 hit every beat at 150 BPM
        beat = 60 / 150
        msgs, t = [], 0.0
        while t < SECONDS - 0.3:
            msgs.append(Message("note_on", note=36, velocity=110, time=t))
            msgs.append(Message("note_off", note=36, velocity=0, time=t + 0.2))
            t += beat
        y = plugin(msgs, duration=SECONDS, sample_rate=SR, num_channels=2, buffer_size=BLOCK)
        r = analyse("midi_kicks", None, y)
        if np.sqrt(np.mean(np.asarray(y) ** 2)) < 10 ** (-90 / 20):
            r["hard"].append("silent output from note-on messages")
        result["tests"].append(r)
    else:
        for name, x in signals().items():
            plugin.reset()
            y = plugin.process(x, sample_rate=SR, buffer_size=BLOCK, reset=True)
            result["tests"].append(analyse(name, x if name != "silence" else None, y))
except Exception as e:
    result["error"] = f"{type(e).__name__}: {e}"[:600]

for r in result["tests"]:
    result["hard"] += [f'{r["signal"]}: {h}' for h in r["hard"]]
    result["warn"] += [f'{r["signal"]}: {w}' for w in r["warn"]]

(out_dir / "listen.json").write_text(json.dumps(result, indent=1))
print(json.dumps(result, indent=1)[:4000])
sys.exit(1 if (result["hard"] or result["error"]) else 0)
