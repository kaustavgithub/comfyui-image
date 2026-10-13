"""Documentary nodes: photos -> narrated Ken Burns video.

  Documentary Story (Ollama)   a vision model writes one narration per image
  Narrate Storyboard (Kokoro)  local text-to-speech, one clip per image
  Ken Burns Video              pan/zoom per image for exactly its narration,
                               rendered on the GPU and streamed to ffmpeg

The nodes pass a DOC_STORYBOARD between them: a dict with "images" (IMAGE
tensors, [1,H,W,3]), "texts" and, after narration, "audio" (float32 arrays at
"sample_rate").
"""
import base64
import io
import json
import math
import os
import random
import subprocess
import tempfile
import urllib.request
import wave

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import comfy.model_management
import folder_paths
from comfy_api.latest import InputImpl

STORYBOARD = "DOC_STORYBOARD"
MAX_IMAGES = 8
OLLAMA_URL = os.environ.get("DOC_OLLAMA_URL", "http://ollama.ollama.svc.cluster.local:11434")

# Kokoro-82M (Apache-2.0). Weights download once into the models volume.
KOKORO_REPO = "hexgrad/Kokoro-82M"
KOKORO_RATE = 24000
VOICES = ["af_heart", "af_bella", "af_nicole", "af_sarah", "am_michael", "am_fenrir",
          "am_puck", "bf_emma", "bf_isabella", "bm_george", "bm_fable", "bm_lewis"]


def _jpeg_b64(image, max_side=1024):
    arr = (image[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    pil = Image.fromarray(arr)
    pil.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def _collect_images(kwargs):
    return [kwargs[f"image_{i}"] for i in range(1, MAX_IMAGES + 1) if kwargs.get(f"image_{i}") is not None]


class DocumentaryStory:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {f"image_{i}": ("IMAGE",) for i in range(2, MAX_IMAGES + 1)}
        return {
            "required": {
                "image_1": ("IMAGE",),
                "brief": ("STRING", {"multiline": True, "default": "",
                                     "placeholder": "Optional: what the photos are about (place, people, year, mood)"}),
                "model": ("STRING", {"default": "qwen2.5vl:7b"}),
                "language": ("STRING", {"default": "English"}),
                "words_per_image": ("INT", {"default": 45, "min": 10, "max": 200}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.5, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "narration_override": ("STRING", {"multiline": True, "default": "",
                                                  "placeholder": "Optional: your own narration, one paragraph per image, separated by blank lines. Skips the model."}),
            },
            "optional": optional,
        }

    RETURN_TYPES = (STORYBOARD, "STRING")
    RETURN_NAMES = ("storyboard", "story")
    FUNCTION = "run"
    CATEGORY = "documentary"

    def run(self, image_1, brief, model, language, words_per_image, temperature, seed,
            narration_override, **kwargs):
        images = _collect_images({"image_1": image_1, **kwargs})
        n = len(images)
        title = ""
        if narration_override.strip():
            texts = [p.strip() for p in narration_override.strip().split("\n\n") if p.strip()]
            if len(texts) != n:
                raise ValueError(f"narration_override has {len(texts)} paragraphs, but {n} images are connected")
        else:
            title, texts = self._ask_model(images, brief, model, language, words_per_image, temperature, seed)
        story = (f"{title}\n\n" if title else "") + "\n\n".join(f"[{i + 1}] {t}" for i, t in enumerate(texts))
        return ({"images": images, "texts": texts, "title": title}, story)

    def _ask_model(self, images, brief, model, language, words, temperature, seed):
        n = len(images)
        prompt = (
            f"You are the narrator of a short documentary. You get {n} photographs, in order. "
            f"Write a narration that tells one connected story across them, like a documentary voice-over: "
            f"concrete, vivid, calm, in the present tense, no lists. Narrate the world in the pictures, "
            f"never the pictures themselves: no words like photo, image, frame, camera, focus, zoom, background or blur. "
            f"Write exactly {n} segments, one per photograph, in the same order; each about {words} words "
            f"and about what is visible in that photograph, linking naturally to the next. "
            f"Language: {language}. "
            + (f"Background from the person who took the photos: {brief.strip()} " if brief.strip() else "")
            + f'Answer only with JSON: {{"title": "...", "segments": ["...", ...]}} with exactly {n} segments.'
        )
        schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "segments": {"type": "array", "items": {"type": "string"}, "minItems": n, "maxItems": n},
            },
            "required": ["title", "segments"],
        }
        body = {
            "model": model,
            "stream": False,
            "format": schema,
            "options": {"temperature": temperature, "seed": seed},
            "messages": [{"role": "user", "content": prompt, "images": [_jpeg_b64(im) for im in images]}],
        }
        req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                reply = json.load(r)["message"]["content"]
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Ollama ({OLLAMA_URL}) said {e.code}: {e.read().decode(errors='replace')[:500]}")
        data = json.loads(reply)
        texts = [s.strip() for s in data.get("segments", []) if s.strip()]
        if len(texts) != n:
            raise RuntimeError(f"The model wrote {len(texts)} segments for {n} images; run again or change the seed. Reply: {reply[:500]}")
        return data.get("title", "").strip(), texts


_kokoro = {}


def _kokoro_dir():
    return os.path.join(folder_paths.models_dir, "kokoro")


def _kokoro_pipeline(lang_code, device):
    from huggingface_hub import hf_hub_download
    from kokoro import KModel, KPipeline

    if "model" not in _kokoro or _kokoro["device"] != device:
        config = hf_hub_download(KOKORO_REPO, "config.json", local_dir=_kokoro_dir())
        weights = hf_hub_download(KOKORO_REPO, "kokoro-v1_0.pth", local_dir=_kokoro_dir())
        _kokoro.clear()
        _kokoro["model"] = KModel(repo_id=KOKORO_REPO, config=config, model=weights).to(device).eval()
        _kokoro["device"] = device
    key = f"pipeline_{lang_code}"
    if key not in _kokoro:
        _kokoro[key] = KPipeline(lang_code=lang_code, repo_id=KOKORO_REPO, model=_kokoro["model"])
    return _kokoro[key]


class NarrateStoryboard:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "storyboard": (STORYBOARD,),
                "voice": (VOICES, {"default": "bm_george"}),
                "speed": ("FLOAT", {"default": 0.95, "min": 0.5, "max": 2.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = (STORYBOARD, "AUDIO")
    RETURN_NAMES = ("storyboard", "narration")
    FUNCTION = "run"
    CATEGORY = "documentary"

    def run(self, storyboard, voice, speed):
        from huggingface_hub import hf_hub_download

        device = comfy.model_management.get_torch_device()
        pipeline = _kokoro_pipeline(voice[0], device)
        voice_path = hf_hub_download(KOKORO_REPO, f"voices/{voice}.pt", local_dir=_kokoro_dir())
        clips = []
        for text in storyboard["texts"]:
            parts = [r.audio.detach().cpu().numpy() for r in pipeline(text, voice=voice_path, speed=speed)
                     if r.audio is not None]
            if not parts:
                raise RuntimeError(f"Kokoro produced no audio for: {text[:80]}")
            clips.append(np.concatenate(parts).astype(np.float32))
        full = np.concatenate(clips)
        audio = {"waveform": torch.from_numpy(full)[None, None, :], "sample_rate": KOKORO_RATE}
        return ({**storyboard, "audio": clips, "sample_rate": KOKORO_RATE}, audio)


def _ease(t):
    return 0.5 - 0.5 * math.cos(math.pi * t)


def _cover(image, width, height, zoom, device):
    """Image as [1,3,h,w] on device, cropped to the output's aspect ratio and
    scaled (with antialiasing) to the output size times the maximum zoom."""
    img = image[0].permute(2, 0, 1)[None].to(device, torch.float32)
    _, _, h, w = img.shape
    target = width / height
    if w / h > target:
        cw = round(h * target)
        x0 = (w - cw) // 2
        img = img[:, :, :, x0:x0 + cw]
    else:
        ch = round(w / target)
        y0 = (h - ch) // 2
        img = img[:, :, y0:y0 + ch, :]
    size = (round(height * zoom), round(width * zoom))
    return F.interpolate(img, size=size, mode="bicubic", antialias=True, align_corners=False).clamp(0, 1)


def _motion(index, mode, zoom, rng):
    """Start and end (scale, x, y): scale >= 1 is the zoom, x/y in [-1, 1] the
    pan within the room the zoom leaves."""
    if mode == "alternate":
        mode = ["zoom in", "zoom out", "pan right", "pan left"][index % 4]
    elif mode == "random":
        mode = rng.choice(["zoom in", "zoom out", "pan right", "pan left"])
    jx, jy = rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6)
    mid = 1 + (zoom - 1) * 0.6
    return {
        "zoom in": ((1.0, 0.0, 0.0), (zoom, jx, jy)),
        "zoom out": ((zoom, jx, jy), (1.0, 0.0, 0.0)),
        "pan right": ((mid, -1.0, jy), (mid, 1.0, -jy)),
        "pan left": ((mid, 1.0, jy), (mid, -1.0, -jy)),
    }[mode]


def _frame(base, scale, x, y, width, height):
    # Window of 1/scale of the image, centred so that x/y = +-1 reaches the edge.
    room = 1 - 1 / scale
    theta = torch.tensor([[1 / scale, 0, x * room], [0, 1 / scale, y * room]],
                         dtype=torch.float32, device=base.device)[None]
    grid = F.affine_grid(theta, (1, 3, height, width), align_corners=False)
    return F.grid_sample(base, grid, mode="bicubic", padding_mode="border", align_corners=False).clamp(0, 1)


def _write_wav(path, samples, rate):
    pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


class KenBurnsVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "storyboard": (STORYBOARD,),
                "width": ("INT", {"default": 1920, "min": 256, "max": 3840, "step": 2}),
                "height": ("INT", {"default": 1080, "min": 256, "max": 2160, "step": 2}),
                "fps": ("INT", {"default": 30, "min": 10, "max": 60}),
                "zoom": ("FLOAT", {"default": 1.2, "min": 1.0, "max": 2.0, "step": 0.05}),
                "motion": (["alternate", "random", "zoom in", "zoom out", "pan right", "pan left"],),
                "pause_before": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 5.0, "step": 0.1,
                                           "tooltip": "Seconds of image before each narration starts"}),
                "pause_after": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1,
                                          "tooltip": "Seconds of image after each narration ends"}),
                "fade": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.1,
                                   "tooltip": "Fade through black between images (seconds)"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "run"
    CATEGORY = "documentary"

    def run(self, storyboard, width, height, fps, zoom, motion, pause_before, pause_after, fade, seed):
        if "audio" not in storyboard:
            raise ValueError("Connect the storyboard from 'Narrate Storyboard' (it has no narration yet)")
        rate = storyboard["sample_rate"]
        rng = random.Random(seed)
        device = comfy.model_management.get_torch_device()

        # Each segment lasts its narration plus the pauses, rounded to whole
        # frames; the audio is padded to exactly the same length, so they stay in sync.
        segments, audio = [], []
        for clip in storyboard["audio"]:
            frames = math.ceil((pause_before + len(clip) / rate + pause_after) * fps)
            total = round(frames / fps * rate)
            lead = round(pause_before * rate)
            seg = np.zeros(total, dtype=np.float32)
            seg[lead:lead + len(clip)] = clip[:total - lead]
            segments.append(frames)
            audio.append(seg)

        out_dir = folder_paths.get_temp_directory()
        os.makedirs(out_dir, exist_ok=True)
        fd, video_path = tempfile.mkstemp(prefix="documentary_", suffix=".mp4", dir=out_dir)
        os.close(fd)
        with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
            wav = os.path.join(tmp, "narration.wav")
            _write_wav(wav, np.concatenate(audio), rate)
            cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                   "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
                   "-i", wav,
                   "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
                   "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", video_path]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                with torch.no_grad():
                    for index, (image, frames) in enumerate(zip(storyboard["images"], segments)):
                        base = _cover(image, width, height, zoom, device)
                        (s0, x0, y0), (s1, x1, y1) = _motion(index, motion, zoom, rng)
                        fade_frames = round(fade / 2 * fps)
                        for f in range(frames):
                            comfy.model_management.throw_exception_if_processing_interrupted()
                            t = _ease(f / max(frames - 1, 1))
                            frame = _frame(base, s0 + (s1 - s0) * t, x0 + (x1 - x0) * t,
                                           y0 + (y1 - y0) * t, width, height)
                            if fade_frames:
                                edge = min(f + 1, frames - f)
                                if edge <= fade_frames:
                                    frame = frame * (edge / (fade_frames + 1))
                            rgb = (frame[0].permute(1, 2, 0) * 255).round().to(torch.uint8).cpu().numpy()
                            proc.stdin.write(rgb.tobytes())
                        del base
            except BrokenPipeError:
                pass
            finally:
                proc.stdin.close()
                err = proc.stderr.read().decode(errors="replace")
                if proc.wait() != 0:
                    raise RuntimeError(f"ffmpeg failed: {err[-1000:]}")
        return (InputImpl.VideoFromFile(video_path),)


NODE_CLASS_MAPPINGS = {
    "DocumentaryStory": DocumentaryStory,
    "DocumentaryNarrate": NarrateStoryboard,
    "DocumentaryKenBurnsVideo": KenBurnsVideo,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DocumentaryStory": "Documentary Story (Ollama)",
    "DocumentaryNarrate": "Narrate Storyboard (Kokoro)",
    "DocumentaryKenBurnsVideo": "Ken Burns Video",
}
