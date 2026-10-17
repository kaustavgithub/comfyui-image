"""Documentary nodes: photos -> narrated Ken Burns video.

  Documentary Story (Ollama)   3-10 images, each with an optional note: a vision
                               model identifies the subject and what it does, a
                               text model writes the narration in the chosen mood
  Narrate Storyboard           local text-to-speech, one clip per image: Chatterbox
                               (expressive, voice and emotion from the mood) or Kokoro
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
MIN_IMAGES = 3
MAX_IMAGES = 10
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


def _collect(kwargs):
    """(image, note) for each connected image, in input order."""
    out = []
    for i in range(1, MAX_IMAGES + 1):
        image = kwargs.get(f"image_{i}")
        if image is not None:
            out.append((image, str(kwargs.get(f"note_{i}") or "").strip()))
    return out


# Narration moods: what the writer is asked to sound like. "custom" uses only
# the custom_style text (which, when filled, also overrides any other mood).
MOODS = {
    "documentary (Nat Geo)": (
        "A National Geographic wildlife documentary narrator: warm, curious, with a dry sense of humour. "
        "Tell the life story behind each animal: how it survives, hunts, migrates, raises young or gets through "
        "the season, and one surprising true fact, tied to what it is doing at that moment. When it is doing "
        "nothing in particular, end with a funny one-liner, like a narrator who has watched it for far too long."
    ),
    "funny": (
        "A stand-up comedian narrating someone's nature slideshow: cheeky, quick and kind. Give the subjects "
        "inner monologues, petty rivalries and human problems (rent, Mondays, group chats, diets). Build a running "
        "gag that comes back across the moments, and end every moment on a punchline."
    ),
    "poetic": (
        "A lyrical, contemplative voice-over: slow, gentle, full of fresh imagery. Speak about what the moment "
        "means: time, seasons, patience, belonging, impermanence. Each moment ends on a quiet, resonant line."
    ),
    "bedtime story": (
        "A bedtime story for young children: soft, simple words, short sentences, cosy and full of wonder. "
        "Give the animals friendly names and make them characters in one small adventure that continues from "
        "moment to moment (they meet, help each other, find something). The last moment winds down to sleep."
    ),
    "epic trailer": (
        "The voice-over of a blockbuster movie trailer for an imagined film starring these creatures. "
        "Pitch the film, don't describe the shots. Give it a real plot: a named hero with a concrete goal (for "
        "example, to survive the winter or win back a territory), a concrete rival or danger, allies, a twist, "
        "and a climax; each moment moves that plot on. Short, punchy, dramatic lines and fragments ('In a world "
        "where…', 'This winter…'), but concrete, not vague abstractions. The last moment ends with the film's "
        "title and a tagline."
    ),
    "nostalgic memoir": (
        "A first-person memoir, years later, by the person who took the photos: warm, reflective, a little "
        "wistful. Talk about the trip and the feeling of being there: the cold mornings, the waiting, the "
        "people, what that moment meant to you, and what has changed since."
    ),
    "sports commentary": (
        "Live sports commentary of a championship final: fast, excited play-by-play. The animals are star "
        "athletes with records and rivalries; give stats, tactics, replays and a running score across the "
        "moments, building to a final whistle and a winner."
    ),
    "mystery / noir": (
        "A hard-boiled film-noir detective voice-over, first person: dry, world-weary, full of wry similes. "
        "The animals are suspects, witnesses and the accused in one case that unfolds across the moments, with "
        "clues, a twist and a verdict at the end."
    ),
    "custom": "",
}
DEFAULT_MOOD = "documentary (Nat Geo)"

# How each mood sounds in Narrate Storyboard. voice: a Kokoro voice (Chatterbox
# copies it from a reference clip); intensity, cfg: Chatterbox's exaggeration
# and cfg_weight (lower cfg = faster, freer delivery); stretch: pace (<1 slower);
# semitones: pitch shift; kokoro_speed: pace with the Kokoro engine.
# "epic trailer" is your pick K (2026-10-14).
VOICE_PRESETS = {
    "documentary (Nat Geo)": {"voice": "bm_george", "intensity": 0.5, "cfg": 0.5, "stretch": 1.0, "semitones": 0, "kokoro_speed": 0.95},
    "funny": {"voice": "am_puck", "intensity": 0.8, "cfg": 0.4, "stretch": 1.0, "semitones": 0, "kokoro_speed": 1.05},
    "poetic": {"voice": "bf_emma", "intensity": 0.45, "cfg": 0.55, "stretch": 0.92, "semitones": 0, "kokoro_speed": 0.85},
    "bedtime story": {"voice": "af_heart", "intensity": 0.35, "cfg": 0.6, "stretch": 0.95, "semitones": 0, "kokoro_speed": 0.85},
    "epic trailer": {"voice": "af_heart", "intensity": 1.0, "cfg": 0.4, "stretch": 0.92, "semitones": -1, "kokoro_speed": 0.85},
    "nostalgic memoir": {"voice": "bf_isabella", "intensity": 0.5, "cfg": 0.5, "stretch": 0.95, "semitones": 0, "kokoro_speed": 0.9},
    "sports commentary": {"voice": "am_michael", "intensity": 1.4, "cfg": 0.3, "stretch": 1.0, "semitones": 0, "kokoro_speed": 1.15},
    "mystery / noir": {"voice": "bm_lewis", "intensity": 0.6, "cfg": 0.4, "stretch": 0.95, "semitones": -1, "kokoro_speed": 0.9},
}
AUTO_VOICE = "auto (from mood)"

LOOK_PROMPT = (
    "Look at this photograph like a careful observer. Identify the main subject as precisely as you can: "
    "an animal or plant by its common species name (if unsure between close species, name the likeliest), "
    "a place or landmark by name if you recognise it, people by what they are doing (never guess who they are). "
    "Say what it is doing right now, and what it holds in its beak, mouth, hands or claws, if anything. "
    "Describe the ground or setting it is in (for example grass, water, rock, sky, street, room), "
    "and list other living things in the scene. Facts only, no style. "
    "Background from the person who took the photos: {brief} "
    "{note}"
    'Answer only with JSON: {{"subject": "...", "activity": "...", "holding": "...", "setting": "...", "others": "..."}}'
)
LOOK_FIELDS = ("subject", "activity", "holding", "setting", "others")
LOOK_SCHEMA = {
    "type": "object",
    "properties": {k: {"type": "string"} for k in LOOK_FIELDS},
    "required": list(LOOK_FIELDS),
}


def _ollama_json(model, content, schema, temperature, seed, images=None):
    message = {"role": "user", "content": content}
    if images:
        message["images"] = images
    body = {"model": model, "stream": False, "format": schema, "think": False,
            "options": {"temperature": temperature, "seed": seed}, "messages": [message]}
    req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            reply = json.load(r)["message"]["content"]
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Ollama ({OLLAMA_URL}, {model}) said {e.code}: {e.read().decode(errors='replace')[:500]}")
    try:
        return json.loads(reply)
    except json.JSONDecodeError:
        raise RuntimeError(f"{model} did not answer with JSON; run again or change the seed. Reply: {reply[:500]}")


def _reference_facts(subject, limit=1200):
    """The intro of the subject's English Wikipedia article (via search), so the
    writer has true facts to use instead of inventing them. "" if not found."""
    import urllib.parse

    if not subject or subject.lower() in ("none", "unknown", "nothing"):
        return ""
    headers = {"User-Agent": "comfyui-documentary/1.0 (homelab)"}
    api = "https://en.wikipedia.org/w/api.php?"
    try:
        q = urllib.parse.urlencode({"action": "query", "list": "search", "srsearch": subject, "srlimit": 1,
                                    "format": "json"})
        with urllib.request.urlopen(urllib.request.Request(api + q, headers=headers), timeout=15) as r:
            hits = json.load(r)["query"]["search"]
        if not hits:
            return ""
        q = urllib.parse.urlencode({"action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1,
                                    "redirects": 1, "titles": hits[0]["title"], "format": "json"})
        with urllib.request.urlopen(urllib.request.Request(api + q, headers=headers), timeout=15) as r:
            pages = json.load(r)["query"]["pages"]
        text = next(iter(pages.values())).get("extract", "").strip()
        return f"{hits[0]['title']}: {text[:limit]}" if text else ""
    except (urllib.error.URLError, OSError, KeyError, ValueError, StopIteration):
        return ""


def _ollama_unload(model):
    # Ollama keeps a model loaded for minutes; a big writer spans both GPUs and
    # would leave the narration and video steps without memory on GPU 0.
    body = {"model": model, "keep_alive": 0}
    req = urllib.request.Request(f"{OLLAMA_URL}/api/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=60).read()
    except (urllib.error.URLError, OSError):
        pass


class DocumentaryStory:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {f"image_{i}": ("IMAGE",) for i in range(MIN_IMAGES + 1, MAX_IMAGES + 1)}
        for i in range(1, MAX_IMAGES + 1):
            optional[f"note_{i}"] = ("STRING", {
                "default": "", "placeholder": f"Optional: about image {i} (what or who it is, what is happening)",
                "tooltip": f"Your description of image {i}; the narration treats it as true"})
        return {
            "required": {
                **{f"image_{i}": ("IMAGE",) for i in range(1, MIN_IMAGES + 1)},
                "brief": ("STRING", {"multiline": True, "default": "",
                                     "placeholder": "Optional: what the photos are about (place, season, people, year)"}),
                "mood": (list(MOODS), {"default": DEFAULT_MOOD,
                                       "tooltip": "How the narration sounds; 'custom' uses custom_style"}),
                "custom_style": ("STRING", {"multiline": True, "default": "",
                                            "placeholder": "Optional: describe your own tone (e.g. 'a pirate captain telling tall tales'). When filled, used instead of the mood."}),
                "model": ("STRING", {"default": "qwen2.5vl:7b",
                                     "tooltip": "Vision model: identifies the subject and what it does, one image at a time"}),
                "writer_model": ("STRING", {"default": "qwen3:32b",
                                            "tooltip": "Text model that writes the narration from what the vision model saw (qwen3:14b: about twice as fast, flatter). Empty: the vision model writes it"}),
                "language": ("STRING", {"default": "English"}),
                "words_per_image": ("INT", {"default": 45, "min": 10, "max": 200}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.5, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "narration_override": ("STRING", {"multiline": True, "default": "",
                                                  "placeholder": "Optional: your own narration, one paragraph per image, separated by blank lines. Skips the models."}),
            },
            "optional": optional,
        }

    RETURN_TYPES = (STORYBOARD, "STRING")
    RETURN_NAMES = ("storyboard", "story")
    FUNCTION = "run"
    CATEGORY = "documentary"

    def run(self, brief, mood, custom_style, model, writer_model, language, words_per_image, temperature, seed,
            narration_override, **kwargs):
        shots = _collect(kwargs)
        n = len(shots)
        if n < MIN_IMAGES:
            raise ValueError(f"Connect at least {MIN_IMAGES} images (connected: {n})")
        images = [image for image, _ in shots]
        notes = [note for _, note in shots]
        style = custom_style.strip() or MOODS.get(mood, "")
        if not style:
            raise ValueError("Mood 'custom' needs a custom_style")
        title, facts = "", []
        if narration_override.strip():
            texts = [p.strip() for p in narration_override.strip().split("\n\n") if p.strip()]
            if len(texts) != n:
                raise ValueError(f"narration_override has {len(texts)} paragraphs, but {n} images are connected")
        else:
            writer = writer_model.strip() or model
            try:
                facts, texts, title = self._narrate(images, notes, brief, model, writer, language,
                                                    words_per_image, temperature, seed, style)
            finally:
                for m in {model, writer}:
                    _ollama_unload(m)
        story = (f"{title}\n\n" if title else "") + "\n\n".join(
            f"[{i + 1}]" + (f" ({facts[i]['subject']}: {facts[i]['activity']})" if facts else "") + f" {t}"
            for i, t in enumerate(texts))
        return ({"images": images, "texts": texts, "title": title, "mood": mood}, story)

    def _narrate(self, images, notes, brief, vision, writer, language, words, temperature, seed, style):
        # One look per image: with all images in one request the vision model lost
        # track of which image it was on. All looks first, then all writing: the
        # two models may share a GPU, and alternating would reload one each time.
        facts = []
        for image, note in zip(images, notes):
            comfy.model_management.throw_exception_if_processing_interrupted()
            look = LOOK_PROMPT.format(brief=brief.strip() or "none",
                                      note=f"The person who took this photograph says about it (treat as true): {note} " if note else "")
            seen = _ollama_json(vision, look, LOOK_SCHEMA, 0.2, seed, images=[_jpeg_b64(image)])
            facts.append({k: str(seen.get(k, "")).strip() for k in LOOK_FIELDS})
        shots = [{**f, **({"photographer_note": n} if n else {})} for f, n in zip(facts, notes)]
        references = {}
        for shot in shots:
            subject = shot.get("photographer_note") or shot["subject"]
            if subject not in references:
                references[subject] = _reference_facts(shot["subject"] if len(subject) > 60 else subject)
            shot["reference"] = references[subject]
        n = len(images)
        common = (
            f"Tone and approach: {style} "
            + (f"Background from the person who took the photos: {brief.strip()} " if brief.strip() else "")
            + "The viewer sees each picture while the narration plays, so never describe what is visible "
            "(colours, poses, perches, grass, sky, light). Use what is in the pictures only as the starting point, "
            "and give the viewer what they can't see: the story, the stakes, feelings, meaning, jokes, true facts. "
            "Never mention a photo, image, picture, camera, frame, shot, slide or background. "
            "Real-world facts (numbers, habits, abilities, records) may only come from the 'reference' text given "
            "for a moment; never invent or embellish one, and if there is no reference, use no facts. "
            "Don't label facts ('fun fact'). Use common names, never scientific names. "
        )
        # Plan first, so the moments add up to one story instead of n captions.
        plan_prompt = (
            f"You are writing a narrated slideshow of {n} moments, in {language}. {common}"
            f"What an observer saw in each moment, in order: {json.dumps(shots, ensure_ascii=False)}. "
            f"Plan the narration as one story in this tone: a title, a one-sentence premise, and exactly {n} beats, "
            "one per moment in order, each saying what that moment's narration does for the story "
            "(not what the picture shows). "
            'Answer only with JSON: {"title": "...", "premise": "...", "beats": ["...", ...]}'
        )
        plan = _ollama_json(writer, plan_prompt, {
            "type": "object",
            "properties": {"title": {"type": "string"}, "premise": {"type": "string"},
                           "beats": {"type": "array", "items": {"type": "string"}, "minItems": n, "maxItems": n}},
            "required": ["title", "premise", "beats"]}, temperature, seed)
        beats = [str(b).strip() for b in plan.get("beats", [])]
        beats += [""] * (n - len(beats))
        texts = []
        for index, (shot, beat) in enumerate(zip(shots, beats)):
            comfy.model_management.throw_exception_if_processing_interrupted()
            prompt = (
                f"You write the narration for moment {index + 1} of {n} of a narrated slideshow titled "
                f"\"{plan.get('title', '')}\". {common}"
                f"The story's premise: {plan.get('premise', '')} "
                f"This moment's job in the story: {beat} "
                f"What an observer saw in this moment (the viewer sees it too; don't list it): "
                f"{json.dumps(shot, ensure_ascii=False)}. "
                f"Write {words} words at most, to be spoken aloud in {language}. "
                "Vary how sentences start, and never use the phrase 'right now'. "
                + (f"The moments so far were narrated like this; continue the story from them without repeating "
                   f"their phrases, images or jokes: {json.dumps(texts, ensure_ascii=False)}. " if texts else "")
                + 'Answer only with JSON: {"narration": "..."}'
            )
            reply = _ollama_json(writer, prompt, {"type": "object", "properties": {"narration": {"type": "string"}},
                                                  "required": ["narration"]}, temperature, seed + index + 1)
            text = str(reply.get("narration", "")).strip()
            if not text:
                raise RuntimeError(f"{writer} wrote no narration for image {index + 1}; run again or change the seed")
            texts.append(text)
        return facts, texts, str(plan.get("title", "")).strip()


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
                "voice": ([AUTO_VOICE] + VOICES, {"default": AUTO_VOICE,
                                                  "tooltip": "auto: the mood's voice (see Documentary Story)"}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05,
                                    "tooltip": "Pace on top of the mood's own pace (1.0 = as the mood sets it)"}),
                # New inputs go last, so saved workflows keep their widget values.
                "engine": (["chatterbox", "kokoro"], {"default": "chatterbox",
                                                      "tooltip": "chatterbox: expressive, follows the mood; kokoro: fast, calm and even"}),
                "intensity": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                        "tooltip": "Chatterbox emotion intensity; 0 = as the mood sets it (0.35 calm ... 1.4 very excited)"}),
            },
        }

    RETURN_TYPES = (STORYBOARD, "AUDIO")
    RETURN_NAMES = ("storyboard", "narration")
    FUNCTION = "run"
    CATEGORY = "documentary"

    def run(self, storyboard, voice, speed, engine="kokoro", intensity=0.0):
        preset = {**VOICE_PRESETS[DEFAULT_MOOD], **VOICE_PRESETS.get(storyboard.get("mood"), {})}
        if voice != AUTO_VOICE:
            preset["voice"] = voice
        if intensity > 0:
            preset["intensity"] = intensity
        device = comfy.model_management.get_torch_device()
        if engine == "kokoro":
            clips = _kokoro_clips(storyboard["texts"], preset["voice"], preset["kokoro_speed"] * speed, device)
        else:
            clips = _chatterbox_clips(storyboard["texts"], preset, speed, device)
        full = np.concatenate(clips)
        audio = {"waveform": torch.from_numpy(full)[None, None, :], "sample_rate": KOKORO_RATE}
        return ({**storyboard, "audio": clips, "sample_rate": KOKORO_RATE}, audio)


def _kokoro_clips(texts, voice, speed, device):
    from huggingface_hub import hf_hub_download

    pipeline = _kokoro_pipeline(voice[0], device)
    voice_path = hf_hub_download(KOKORO_REPO, f"voices/{voice}.pt", local_dir=_kokoro_dir())
    clips = []
    for text in texts:
        comfy.model_management.throw_exception_if_processing_interrupted()
        parts = [r.audio.detach().cpu().numpy() for r in pipeline(text, voice=voice_path, speed=speed)
                 if r.audio is not None]
        if not parts:
            raise RuntimeError(f"Kokoro produced no audio for: {text[:80]}")
        clips.append(np.concatenate(parts).astype(np.float32))
    return clips


# Chatterbox (Resemble AI, MIT): expressive TTS with an emotion-intensity
# control ("exaggeration"). It takes its voice from a reference clip; we make
# that clip with Kokoro, so the same voice names work for both engines.
_chatterbox = {}
REFERENCE_TEXT = ("The river bends through the valley, past the old stone bridge, where the morning light "
                  "falls softly on the water and the reeds lean in the wind.")


def _chatterbox_model(device):
    os.environ.setdefault("HF_HOME", os.path.join(folder_paths.models_dir, "chatterbox", "hf"))
    from chatterbox.tts import ChatterboxTTS

    if "model" not in _chatterbox:
        _chatterbox["model"] = ChatterboxTTS.from_pretrained(device=str(device))
    model = _chatterbox["model"]
    _chatterbox_move(model, device)
    return model


def _chatterbox_move(model, device):
    # ComfyUI doesn't manage this model: park it in RAM between runs so its
    # ~3.4 GB on GPU 0 are free for other models.
    for part in ("t3", "s3gen", "ve"):
        getattr(model, part).to(device)
    if model.conds is not None:
        model.conds = model.conds.to(device)
    model.device = str(device)


def _reference_clip(voice, device):
    path = os.path.join(folder_paths.models_dir, "chatterbox", "voices", f"{voice}.wav")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        clip = np.concatenate(_kokoro_clips([REFERENCE_TEXT], voice, 0.9, device))
        _write_wav(path, clip, KOKORO_RATE)
    return path


def _sentences(text, limit=300):
    """Chunks of whole sentences up to ~limit characters: Chatterbox stops after
    about 40 s of speech per call."""
    import re

    chunks, current = [], ""
    for sentence in re.split(r"(?<=[.!?…])\s+", text.strip()):
        if current and len(current) + len(sentence) + 1 > limit:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


def _chatterbox_clips(texts, preset, speed, device):
    import librosa

    model = _chatterbox_model(device)
    try:
        reference = _reference_clip(preset["voice"], device)
        stretch = preset["stretch"] * speed
        clips = []
        for text in texts:
            parts = []
            for chunk in _sentences(text):
                comfy.model_management.throw_exception_if_processing_interrupted()
                wav = model.generate(chunk, audio_prompt_path=reference, exaggeration=preset["intensity"],
                                     cfg_weight=preset["cfg"]).squeeze(0).cpu().numpy().astype(np.float32)
                parts.append(wav)
            clip = np.concatenate(parts)
            if model.sr != KOKORO_RATE:
                clip = librosa.resample(clip, orig_sr=model.sr, target_sr=KOKORO_RATE)
            if preset["semitones"]:
                clip = librosa.effects.pitch_shift(clip, sr=KOKORO_RATE, n_steps=preset["semitones"])
            if abs(stretch - 1.0) > 1e-3:
                clip = librosa.effects.time_stretch(clip, rate=stretch)
            peak = float(np.abs(clip).max()) or 1.0
            clips.append((clip / peak * 0.9).astype(np.float32))
        return clips
    finally:
        _chatterbox_move(model, "cpu")
        torch.cuda.empty_cache()


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
    "DocumentaryNarrate": "Narrate Storyboard",
    "DocumentaryKenBurnsVideo": "Ken Burns Video",
}
