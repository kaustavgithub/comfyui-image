"""Documentary nodes: photos -> narrated Ken Burns video.

  Documentary Photo            Load Image plus an optional name and note per photo
  Documentary Story            3-10 images (or photos), each with an optional note:
                               the species from the name, a species named in the
                               note, or BioCLIP 2 (species.py) when it is sure; a
                               vision model says what the subject does, a text
                               model writes the narration in the chosen mood
  Narrate Storyboard           local text-to-speech, one clip per image plus the
                               closing follow line: Chatterbox (expressive, voice and
                               emotion from the mood) or Kokoro
  Documentary Music            instrumental background music for the mood, as long
                               as the narration (ACE-Step 1.5, ComfyUI's own model code)
  Ken Burns Video              intro collage, pan/zoom per image for exactly its
                               narration, transitions, subtitles and a follow card
                               at the end, rendered on the GPU and streamed to
                               ffmpeg; mixes in the music, lower while the narrator speaks
  Social Post                  for the target: an Instagram caption with five hashtags,
                               or a YouTube title, description, tags and three
                               hashtags; plus SEO key phrases
  Documentary Thumbnail        a cover image: the best photo with the story's headline

The nodes pass a DOC_STORYBOARD between them: a dict with "images" (IMAGE
tensors, [1,H,W,3]), "texts", "title", "hook", "outro", "details" (per image
the camera line to show on its slide, or ""), "max_seconds" and, after narration,
"audio" (float32 arrays at "sample_rate"), "timings" (per clip: (text, start,
end) in samples, for subtitles) and "outro_audio"; after music, "music"
([channels, n] at "music_rate").

Menu "Documentary → Stored files" (web/storage.js) lists the files in ComfyUI's
output, input and temp folders and deletes the ones you pick.
"""
import base64
import io
import json
import math
import os
import random
import shutil
import subprocess
import tempfile
import urllib.request
import wave

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import comfy.model_management
import folder_paths
import nodes
from comfy_api.latest import InputImpl

from . import species as species_id

STORYBOARD = "DOC_STORYBOARD"
PHOTO = "DOC_PHOTO"
# The Story's image inputs take a plain IMAGE (Load Image) or a Documentary Photo.
IMAGE_IN = f"IMAGE,{PHOTO}"
MIN_IMAGES = 3
MAX_IMAGES = 10
# The LLM gate (homelab apps/llm): OpenAI-compatible API in front of llama-swap,
# which starts each model on its own GPUs. DOC_LLM_TOKEN is an admin token, so
# these nodes may switch models (the gate refuses that to normal users).
LLM_URL = os.environ.get("DOC_LLM_URL", "http://llm.llm.svc.cluster.local:8080").rstrip("/")
LLM_TOKEN = os.environ.get("DOC_LLM_TOKEN", "")

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
    """(image, name, note, details) for each connected image, in input order. A
    Documentary Photo brings its own name, note and camera details; a plain
    image has none."""
    out = []
    for i in range(1, MAX_IMAGES + 1):
        item = kwargs.get(f"image_{i}")
        if item is None:
            continue
        name, details, note = "", "", str(kwargs.get(f"note_{i}") or "").strip()
        if isinstance(item, dict):
            name, details = item.get("name", ""), item.get("details", "")
            note = " ".join(x for x in (item.get("note", ""), note) if x)
            item = item["image"]
        out.append((item, name, note, details))
    return out


class DocumentaryPhoto:
    """Load Image plus what you know about the photo: the subject's name (used
    as true, and to look up its facts) and a note."""

    @classmethod
    def INPUT_TYPES(cls):
        base = folder_paths.get_input_directory()
        files = [f for f in os.listdir(base) if os.path.isfile(os.path.join(base, f))]
        return {
            "required": {
                "image": (sorted(folder_paths.filter_files_content_types(files, ["image"])), {"image_upload": True}),
                "name": ("STRING", {"default": "", "placeholder": "Optional: what or who it is (e.g. Eurasian blue tit)",
                                    "tooltip": "The subject's name. Used as true instead of the vision model's guess, "
                                               "and to look up its facts"}),
                "note": ("STRING", {"multiline": True, "default": "",
                                    "placeholder": "Optional: what is happening, where, when"}),
                # New inputs go last, so saved workflows keep their widget values.
                "details": ("STRING", {"default": "",
                                       "placeholder": "Optional: camera settings (e.g. Nikon Z6 · 500 mm · f/5.6 · 1/2000 s · ISO 800)",
                                       "tooltip": "Shown small on the photo's slide when the story says it fits"}),
            },
        }

    RETURN_TYPES = (PHOTO, "IMAGE")
    RETURN_NAMES = ("photo", "image")
    FUNCTION = "run"
    CATEGORY = "documentary"

    def run(self, image, name, note, details=""):
        loaded = nodes.LoadImage().load_image(image)[0]
        name = name.strip()
        if name.lower() == "image":
            # Left over from the first template, whose widget values were one off.
            name = ""
        return ({"image": loaded, "name": name, "note": note.strip(), "details": details.strip()}, loaded)

    @classmethod
    def IS_CHANGED(cls, image, **kwargs):
        return nodes.LoadImage.IS_CHANGED(image)

    @classmethod
    def VALIDATE_INPUTS(cls, image, **kwargs):
        return nodes.LoadImage.VALIDATE_INPUTS(image)


# Narration moods: what the writer is asked to sound like. "custom" uses only
# the custom_style text (which, when filled, also overrides any other mood).
MOODS = {
    "documentary (Nat Geo)": (
        "A National Geographic wildlife documentary narrator: warm, curious, with a dry sense of humour. "
        "Tell the life story behind each animal: how it survives, hunts, migrates, raises young or gets through "
        "the season, and, when the moment has a 'reference', one surprising fact from it, tied to what it is doing. When it is doing "
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
        "athletes with rivalries; give tactics, replays and a running score across the moments (the score is "
        "part of the game, not a fact), building to a final whistle and a winner."
    ),
    "mystery / noir": (
        "A hard-boiled film-noir detective voice-over, first person: dry, world-weary, full of wry similes. "
        "The animals are suspects, witnesses and the accused in one case that unfolds across the moments, with "
        "clues, a twist and a verdict at the end."
    ),
    "behind the lens": (
        "The photographer, first person, telling a friend how each shot happened: present tense, warm, honest, "
        "plain everyday words, like talking, not writing. The waiting, how far away or close the animal was, the "
        "light at that hour, the split second it happened, and what it felt like to get it. Use only what the "
        "'camera' settings and the photographer's note say about the shot (the hour, the lens, the shutter "
        "speed, the date); never invent where you were, who you were with, the weather or what happened before. "
        "Say the time of day the way people do ('just after eight in the morning', 'late evening'), never as "
        "digits or 'eight five'. Mention a camera setting only where it is the story (a fast shutter for a "
        "bird that wouldn't sit still, a long lens for one that was far away), at most twice in the whole film. "
        "When a moment has a 'reference', one interesting fact from it."
    ),
    # Reel formats with a fixed job per photo (SHOT_LISTS).
    "mini-documentary": (
        "A calm, observant, grounded and atmospheric wildlife documentary voice, in the style of BBC Earth: "
        "unhurried, precise, quietly in awe. No social media slang, no exclamation marks."
    ),
    "photographer's craft": (
        "A wildlife photographer on the craft: authentic, technical yet accessible, appreciative of both the craft "
        "and nature. Plain words for technique (the light, the soft background, sharp focus, the catchlight in "
        "the eye, timing); a camera setting only when it is the point."
    ),
    "species spotlight": (
        "A visual curator presenting a gallery of birds: rhythmic, brisk and engaging, each line a crisp "
        "spotlight on one bird."
    ),
    "custom": "",
}
# Moods with a fixed job for each photo, in order, and the words per photo
# (least, most); "camera": the writer sees the camera settings; "ask": the
# last narration must say one of these words, else it is rewritten, and
# finally gets "ask_line". For a number of photos other than the list's, the
# first and last jobs stay, the middle ones are spread. Never picked by auto:
# they are made for reels.
SHOT_LISTS = {
    "mini-documentary": {"words": (8, 12), "jobs": [
        "The hook: set the scene, the weather or the mood, and the patience waiting in the wild takes.",
        "The discovery: spotting the subject, and how it meets the light or the foliage around it.",
        "The detail: one behavioural detail, the texture of its feathers, or its posture.",
        "The climax: the most striking thing in this moment (eye contact, the sharpest focus or its peak pose).",
        "The sign-off: a poetic closing thought about capturing fleeting moments in nature.",
    ]},
    "photographer's craft": {"words": (8, 12), "camera": True, "ask": ("prefer", "favourite", "favorite"),
                             "ask_line": "Which shot do you prefer?", "jobs": [
        "The hook: the challenge of wildlife photography (fleeting light, tiny windows of opportunity).",
        "Framing: the setting, the depth of field (the soft background blur), or how the subject separates "
        "from the background.",
        "Sharpness: the tack-sharp focus, the catchlight in its eye, or fine feather detail.",
        "Timing: the mood, the colour palette, or the split-second timing of the pose.",
        "The sign-off: why patience in the field pays off, then end by asking the viewer which shot they "
        "prefer (say 'prefer').",
    ]},
    "species spotlight": {"words": (4, 10), "ask": ("favourite", "favorite"),
                          "ask_line": "Comment your favourite, 1 to {n}.", "jobs": [
        "The hook: a bold statement about the beauty of birds, introducing this first bird's standout feature.",
        "Contrast: move on smoothly by contrasting this bird's colour, posture or mood with the one before.",
        "Presence: the intensity, the stillness or the surroundings of this bird.",
        "Character: the light on this bird and its distinct character.",
        "Call to action: this last bird in a few words, then end by asking the viewer to comment their "
        "favourite, from 1 to {n} (say 'favourite').",
    ]},
}
# The camera moves the writer chooses from, per photo (Ken Burns Video's
# motion STORY_MOTION uses them).
MOVES = ["zoom in", "zoom out", "pan right", "pan left"]
STORY_MOTION = "auto (from story)"


def _shot_jobs(shot_list, n):
    """The list's jobs for n photos: first and last as they are, the middle ones spread."""
    jobs = [j.replace("{n}", str(n)) for j in shot_list["jobs"]]
    middle = jobs[1:-1]
    return [jobs[0]] + [middle[min(len(middle) - 1, i * len(middle) // (n - 2))] for i in range(n - 2)] + [jobs[-1]]
DEFAULT_MOOD = "documentary (Nat Geo)"
# The photographer's own story of each shot: the writer also sees the camera settings.
LENS_MOOD = "behind the lens"
# Not a mood of its own: the writer picks one of MOODS from what the vision model saw.
AUTO_MOOD = "auto (from photos)"
# Camera details (Documentary Photo's "details") on the slides.
CAMERA_DETAILS = ["auto (the writer decides)", "always", "never"]
# Species identification (species.py).
SPECIES_ID = ["BioCLIP 2", "off"]
# What the vision model is told when the photo shows an animal whose species
# nobody is sure of: a wrong species would bring the wrong facts.
UNSURE_SPECIES = ("Nobody is sure of this animal's species: in 'subject', name only the general kind (for example "
                  "'a duck', 'a small songbird', 'a bird of prey', 'a deer'), never a species. ")
# The small line under the title in the intro, and on the thumbnail.
MOOD_LABELS = {
    "documentary (Nat Geo)": "A wildlife story",
    "funny": "A comedy",
    "poetic": "A visual poem",
    "bedtime story": "A bedtime story",
    "epic trailer": "Coming soon",
    "nostalgic memoir": "A memoir",
    "sports commentary": "The final",
    "mystery / noir": "A case file",
    "behind the lens": "Behind the shot",
    "mini-documentary": "The story behind the shot",
    "photographer's craft": "Behind the lens",
    "species spotlight": "Spotlight",
}
# Spoken at the end when the story has no closing line of its own.
DEFAULT_OUTRO = "If you enjoyed this story, follow for more."

# Where the video goes. A reel is 9:16 and at most 90 s long: the story writes
# fewer words, Narrate speeds up a narration that is still too long, and Ken
# Burns Video uses short intro, pauses, transitions and follow card.
TARGETS = ["YouTube video", "Instagram reel"]
REEL = "Instagram reel"
REEL_SIZE = (1080, 1920)
REEL_MAX = 90.0  # also what the story's word count aims at without max_seconds
REEL_INTRO, REEL_OUTRO, REEL_PAUSES, REEL_FADE = 2.5, 3.0, (0.25, 0.45), 0.5
WORDS_PER_SECOND = 2.1  # how fast the narrators speak, measured
# A reel's budget: Chatterbox spoke a reel at about 3.3 words a second
# (2026-10-25); 2.1 left a 50 s reel at 35 s, in clipped sentences.
REEL_WORDS_PER_SECOND = 2.8


def _is_reel(storyboard):
    return storyboard.get("target") == REEL


def _reel_words(n, limit=0.0):
    """Words per photo for a reel with n photos: a few seconds under `limit`
    (REEL_MAX without one); at most 40, so a few photos make a shorter reel."""
    aim = (min(limit, REEL_MAX) if limit > 0 else REEL_MAX) - 3.0
    speech = aim - REEL_INTRO - (REEL_OUTRO + 0.5) - n * sum(REEL_PAUSES)
    return max(5, min(40, int(speech * REEL_WORDS_PER_SECOND / n)))


# A YouTube video's timing with Ken Burns Video's defaults: intro, pause before
# and after each narration, follow card (at least the closing line + tail).
VIDEO_INTRO, VIDEO_PAUSES, VIDEO_OUTRO, VIDEO_TAIL = 4.5, (0.6, 1.0), 5.0, 1.6


def _video_words(n, limit, words, outro_words=20):
    """Words per photo so a YouTube video with n photos stays under `limit`
    seconds (at most `words`), with a few seconds to spare."""
    closing = outro_words / WORDS_PER_SECOND + VIDEO_TAIL
    speech = limit - 5.0 - VIDEO_INTRO - max(VIDEO_OUTRO, closing) - n * sum(VIDEO_PAUSES)
    return max(10, min(words, int(speech * WORDS_PER_SECOND / n)))


def _limit(storyboard):
    """The longest the video may be, in seconds (0: no limit)."""
    limit = float(storyboard.get("max_seconds") or 0)
    if _is_reel(storyboard):
        return min(limit, REEL_MAX) if limit > 0 else REEL_MAX
    return limit

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
    "behind the lens": {"voice": "bm_fable", "intensity": 0.45, "cfg": 0.55, "stretch": 1.0, "semitones": 0, "kokoro_speed": 1.0},
    "mini-documentary": {"voice": "bm_george", "intensity": 0.4, "cfg": 0.55, "stretch": 0.97, "semitones": 0, "kokoro_speed": 0.92},
    "photographer's craft": {"voice": "bm_fable", "intensity": 0.5, "cfg": 0.5, "stretch": 1.0, "semitones": 0, "kokoro_speed": 1.0},
    "species spotlight": {"voice": "af_bella", "intensity": 0.7, "cfg": 0.45, "stretch": 1.0, "semitones": 0, "kokoro_speed": 1.05},
}
AUTO_VOICE = "auto (from mood)"
# Your own voice (my_voice): at most this much expression, at least this cfg.
MY_VOICE_INTENSITY, MY_VOICE_CFG = 0.4, 0.6

LOOK_PROMPT = (
    "Look at this photograph like a careful observer. Identify the main subject as precisely as you can: "
    "an animal or plant by its common species name (if unsure between close species, name the likeliest), "
    "a place or landmark by name if you recognise it, people by what they are doing (never guess who they are). "
    "Say what it is doing right now, and what it holds in its beak, mouth, hands or claws, if anything. "
    "Describe the ground or setting it is in (for example grass, water, rock, sky, street, room), "
    "and list other living things in the scene. "
    "Say where the main subject sits in the frame, how much of it it fills and which way it faces (for example "
    "'small, in the left third, facing right', 'fills the frame, centred, facing the camera'). Facts only, no style. "
    "Background from the person who took the photos: {brief} "
    "{note}"
    'Answer only with JSON: {{"subject": "...", "activity": "...", "holding": "...", "setting": "...", "others": "...", '
    '"framing": "..."}}'
)
LOOK_FIELDS = ("subject", "activity", "holding", "setting", "others", "framing")
LOOK_SCHEMA = {
    "type": "object",
    "properties": {k: {"type": "string"} for k in LOOK_FIELDS},
    "required": list(LOOK_FIELDS),
}


def _llm_request(path, body=None, timeout=900):
    headers = {"Content-Type": "application/json"}
    if LLM_TOKEN:
        headers["Authorization"] = f"Bearer {LLM_TOKEN}"
    req = urllib.request.Request(f"{LLM_URL}{path}", data=json.dumps(body).encode() if body is not None else None,
                                 headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _llm_json(model, content, schema, temperature, seed, images=None, tries=3, max_tokens=3000):
    """The model's answer as JSON (constrained to the schema). A model can get
    stuck repeating itself (seen with qwen2.5vl): the answer is cut at
    max_tokens, and a broken answer is asked again with another seed."""
    parts = content
    if images:
        parts = [{"type": "text", "text": content}] + [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}} for b64 in images]
    reply = ""
    for attempt in range(tries):
        body = {"model": model, "messages": [{"role": "user", "content": parts}],
                "temperature": temperature, "seed": seed + 1000 * attempt, "max_tokens": max_tokens,
                "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}},
                "chat_template_kwargs": {"enable_thinking": False}}
        try:
            reply = _llm_request("/v1/chat/completions", body)["choices"][0]["message"].get("content") or ""
        except urllib.error.HTTPError as e:
            error = e.read().decode(errors="replace")[:500]
            raise RuntimeError(f"LLM ({LLM_URL}, {model}) said {e.code}: {error}")
        try:
            return json.loads(reply)
        except json.JSONDecodeError:
            print(f"[documentary] {model}: no valid JSON (try {attempt + 1} of {tries}): {reply[:200]!r}")
    raise RuntimeError(f"{model} did not answer with JSON; run again or change the seed. Reply: {reply[:500]}")


NUMBER_WORDS = ("hundred", "thousand", "million", "billion", "dozen")


def _unsupported_numbers(text, reference):
    """Numbers in a narration that its moment's reference doesn't have: digits
    ("5,000", "12") and big number words ("thousands"). The writer adds them
    even when told not to; each one is an invented fact."""
    import re

    ref = (reference or "").lower().replace(",", "")
    found = []
    for m in re.finditer(r"\d[\d,.]*", text):
        digits = m.group().rstrip(".,").replace(",", "")
        if digits and not re.search(rf"(?<![\d.]){re.escape(digits)}(?![\d])", ref):
            found.append(m.group().rstrip(".,"))
    found += [w for w in NUMBER_WORDS if re.search(rf"\b{w}s?\b", text, re.I) and w not in ref]
    return found


def _drop_sentences_with(text, numbers):
    """The narration without the sentences that hold one of `numbers` (at least one sentence stays)."""
    import re

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    keep = [x for x in sentences if not any(n.lower() in x.lower() for n in numbers)]
    return " ".join(keep or sentences[:1])


def _local_first(load):
    """load() with the models from the volume only, without asking Hugging Face
    for updates; online only when a file isn't there yet (its first use). So a
    workflow runs without the internet, and nothing but a model's name ever
    goes out."""
    from huggingface_hub import constants
    from huggingface_hub.errors import LocalEntryNotFoundError, OfflineModeIsEnabled

    was = constants.HF_HUB_OFFLINE
    constants.HF_HUB_OFFLINE = True
    try:
        return load()
    except Exception as e:
        # Some libraries wrap the hub's error in their own.
        if not isinstance(e, (LocalEntryNotFoundError, OfflineModeIsEnabled)) and "offline mode" not in str(e):
            raise
        print(f"[documentary] a model file is not on the volume yet: downloading it once ({str(e)[:120]})")
    finally:
        constants.HF_HUB_OFFLINE = was
    return load()


def _free_gpu(timeout=90):
    """Unload the LLMs on ComfyUI's GPU (e.g. a chat model from Open WebUI, or
    this run's own writer), so the next step has the GPU; they reload on their
    next use. Then wait for the GPU's free memory to settle (a model's memory
    is released a moment after it stops)."""
    import time

    if not torch.cuda.is_available():
        return
    device = comfy.model_management.get_torch_device()
    # Indexes match the gate's: ComfyUI runs with CUDA_DEVICE_ORDER=PCI_BUS_ID.
    gpu = device.index if device.index is not None else torch.cuda.current_device()
    try:
        done = _llm_request(f"/gate/unload?gpu={gpu}", {}, timeout=timeout).get("unloaded", [])
        if done:
            print(f"[documentary] unloaded from GPU {gpu}: {', '.join(done)}")
    except (urllib.error.URLError, OSError, ValueError) as e:
        print(f"[documentary] could not free GPU {gpu} ({e}); going on")
        return
    deadline, last = time.monotonic() + timeout, -1
    while time.monotonic() < deadline:
        free = torch.cuda.mem_get_info(device)[0]
        if free - last < 64 * 1024 ** 2:
            break
        last = free
        time.sleep(1.0)


class DocumentaryStory:
    @classmethod
    def INPUT_TYPES(cls):
        # All optional, so any Photo node can be deleted; run() still wants MIN_IMAGES.
        optional = {f"image_{i}": (IMAGE_IN,) for i in range(1, MAX_IMAGES + 1)}
        for i in range(1, MAX_IMAGES + 1):
            optional[f"note_{i}"] = ("STRING", {
                "default": "", "placeholder": f"Optional: about image {i} (what or who it is, what is happening)",
                "tooltip": f"Your description of image {i}; the narration treats it as true"})
        return {
            "required": {
                "brief": ("STRING", {"multiline": True, "default": "",
                                     "placeholder": "Optional: what the photos are about (place, season, people, year)"}),
                "mood": (list(MOODS) + [AUTO_MOOD], {"default": DEFAULT_MOOD,
                                                     "tooltip": "How the narration sounds; 'custom' uses custom_style; "
                                                                "'auto': the writer picks one that fits the photos"}),
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
                "target": (TARGETS, {"default": TARGETS[0],
                                     "tooltip": "YouTube video: 16:9, words_per_image as set. Instagram reel: 9:16, "
                                                "up to 90 s (or max_seconds; words per image set to fit, at most 40; 4-6 photos work best)"}),
                "max_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1800.0, "step": 5.0,
                                          "tooltip": "The longest the video may be; fewer words per image, and Narrate "
                                                     "speeds up a narration that still runs over. 0 = no limit "
                                                     "(a reel: 90 s)"}),
                "camera_details": (CAMERA_DETAILS, {"default": CAMERA_DETAILS[0],
                                                    "tooltip": "The photos' camera details as a small line on "
                                                               "their slides: where they fit the story, on every "
                                                               "slide that has them, or never"}),
                "species_id": (SPECIES_ID, {"default": SPECIES_ID[0],
                                            "tooltip": "BioCLIP 2 names a bird's or mammal's species when it is "
                                                       "sure (a name given on the photo, or a species named in its "
                                                       "note, always wins). When nobody is sure, the story says "
                                                       "only 'a duck' and uses no facts. off: the vision model "
                                                       "guesses, as before"}),
                "species_min_confidence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                                     "tooltip": "How sure BioCLIP must be to name the species "
                                                                "(its probability, in the log)"}),
                "avoid_moods": ("STRING", {"default": "",
                                           "placeholder": "Optional: moods auto must not pick, comma-separated "
                                                          "(e.g. the last nights')"}),
            },
            "optional": optional,
        }

    RETURN_TYPES = (STORYBOARD, "STRING")
    RETURN_NAMES = ("storyboard", "story")
    FUNCTION = "run"
    CATEGORY = "documentary"

    def run(self, brief, mood, custom_style, model, writer_model, language, words_per_image, temperature, seed,
            narration_override, target=TARGETS[0], max_seconds=0.0, camera_details=CAMERA_DETAILS[0],
            species_id=SPECIES_ID[0], species_min_confidence=0.5, avoid_moods="", **kwargs):
        shots = _collect(kwargs)
        n = len(shots)
        if n < MIN_IMAGES:
            raise ValueError(f"Connect at least {MIN_IMAGES} images (connected: {n})")
        images = [shot[0] for shot in shots]
        names = [shot[1] for shot in shots]
        notes = [shot[2] for shot in shots]
        details = [shot[3] for shot in shots]
        avoid = [m.strip() for m in avoid_moods.split(",") if m.strip()]
        auto = mood == AUTO_MOOD and not custom_style.strip()
        if auto and narration_override.strip():
            mood, auto = DEFAULT_MOOD, False  # nothing to pick from without the models
        style = None if auto else custom_style.strip() or MOODS.get(mood, "")
        if style == "":
            raise ValueError("Mood 'custom' needs a custom_style")
        title, facts, extra = "", [], {}
        reel = target == REEL
        limit = _limit({"target": target, "max_seconds": max_seconds})
        shot_list = None if custom_style.strip() else SHOT_LISTS.get(mood)
        if shot_list:
            words_per_image = shot_list["words"][1]
        elif reel:
            words_per_image = _reel_words(n, max_seconds)
        elif limit > 0:
            words_per_image = _video_words(n, limit, words_per_image)
        if narration_override.strip():
            texts = [p.strip() for p in narration_override.strip().split("\n\n") if p.strip()]
            if len(texts) != n:
                raise ValueError(f"narration_override has {len(texts)} paragraphs, but {n} images are connected")
        else:
            writer = writer_model.strip() or model
            # The models stay loaded for Social Post; Narrate, Music and Video free the GPU.
            known = self._species(images, names, notes, species_id, species_min_confidence)
            facts, texts, title, extra = self._narrate(images, known, notes, details,
                                                       camera_details == CAMERA_DETAILS[0], brief, model, writer,
                                                       language, words_per_image, temperature, seed, style,
                                                       10 if reel else 20, avoid, shot_list)
            mood = extra.get("mood") or mood
        # Camera details: on the slides the writer chose (all of them without the writer), or as set.
        shown = {"always": [True] * n, "never": [False] * n}.get(camera_details) or extra.get("show_details") \
            or [True] * n
        details = [d if d and show else "" for d, show in zip(details, shown)]
        hook, outro = extra.get("hook", ""), extra.get("outro", "")
        motions, cover = extra.get("motions") or [], extra.get("cover")
        story = (f"{title}\n\n" if title else "") + (f"Mood: {mood}\n\n" if auto or shot_list else "") + (
            f"Thumbnail: {hook}" + (f" (photo {cover})" if cover else "") + "\n\n" if hook else "") + "\n\n".join(
            f"[{i + 1}]" + (f" <{motions[i]}>" if i < len(motions) else "")
            + (f" ({facts[i]['subject']}: {facts[i]['activity']})" if facts else "") + f" {t}"
            for i, t in enumerate(texts)) + (f"\n\n[end] {outro}" if outro else "")
        subjects = [f["subject"] for f in facts] if facts else [x for x in names if x]
        return ({"images": images, "texts": texts, "title": title, "hook": hook, "outro": outro, "mood": mood,
                 "custom_style": custom_style.strip(), "brief": brief.strip(), "subjects": subjects,
                 "language": language, "target": target, "max_seconds": limit, "details": details,
                 "motions": motions, "cover": cover}, story)

    @staticmethod
    def _species(images, names, notes, mode, min_confidence):
        """Per image {"name", "facts", "animal"}: the subject's name and its
        facts (the species' Wikipedia intro, from the local species list; ""
        for both when unknown), and whether it is an animal of unsure species
        (None: not known). In order: the name given on the photo, a species
        named in its note, BioCLIP when sure. Nothing leaves the homelab."""
        known = [{"name": name, "facts": "", "animal": None} for name in names]
        if mode == "off":
            for k in known:
                try:
                    k["facts"] = species_id.facts(k["name"])
                except Exception:  # no species list (offline)
                    pass
            return known
        try:
            for k in known:
                if k["name"]:
                    k["facts"] = species_id.facts(k["name"])
            for k, note in zip(known, notes):
                hit = None if k["name"] else species_id.named_in(note)
                if hit:
                    k.update(name=hit[0], facts=species_id.facts(hit[1]), source="your note")
            todo = [i for i, k in enumerate(known) if not k["name"]]
            found = []
            if todo:
                todo_images = [images[i][0] for i in todo]
                try:
                    found = _local_first(lambda: species_id.identify(todo_images,
                                                                     comfy.model_management.get_torch_device()))
                except torch.cuda.OutOfMemoryError:  # the LLMs fill the GPU: slower, on the CPU
                    found = _local_first(lambda: species_id.identify(todo_images, "cpu"))
            for i, picks in zip(todo, found):
                (common, scientific, p), runner = picks[0], picks[1]
                known[i]["best"] = f"{common or scientific} {p:.2f}, then {runner[0] or runner[1]} {runner[2]:.2f}"
                if common is None:
                    known[i]["animal"] = False  # a landscape, a person, a building: the vision model names it
                elif p >= min_confidence:
                    known[i].update(name=common, facts=species_id.facts(scientific), source=f"BioCLIP {p:.2f}")
                else:
                    known[i]["animal"] = True
        except Exception as e:  # no network for GBIF, no weights: the vision model guesses, as before
            print(f"[documentary] species identification failed ({e}); the vision model names the subjects")
            return [{"name": name, "facts": "", "animal": None} for name in names]
        for i, k in enumerate(known, 1):
            said = k["name"] or ("an animal, species unsure" if k["animal"] else "not an animal")
            print(f"[documentary] species {i}: {said} ({k.get('source') or k.get('best') or 'given'})")
        return known

    def _narrate(self, images, known, notes, details, ask_show, brief, vision, writer, language, words,
                 temperature, seed, style, outro_words=20, avoid=(), shot_list=None):
        """ask_show: the plan says on which slides the camera details fit.
        shot_list: one of SHOT_LISTS (each photo's job fixed; the plan also
        picks a camera move per photo and the cover photo)."""
        # One look per image: with all images in one request the vision model lost
        # track of which image it was on. All looks first, then all writing: the
        # two models may share a GPU, and alternating would reload one each time.
        facts = []
        names = [k["name"] for k in known]
        for image, k, note in zip(images, known, notes):
            name = k["name"]
            comfy.model_management.throw_exception_if_processing_interrupted()
            look = LOOK_PROMPT.format(brief=brief.strip() or "none",
                                      note=(f"The main subject is: {name} (given by the person who took the photograph; "
                                            f"it is correct, use exactly this as the subject). " if name else "")
                                      + (UNSURE_SPECIES if k["animal"] else "")
                                      + (f"The person who took this photograph says about it (treat as true): {note} " if note else ""))
            seen = _llm_json(vision, look, LOOK_SCHEMA, 0.2, seed, images=[_jpeg_b64(image)])
            fact = {k: str(seen.get(k, "")).strip() for k in LOOK_FIELDS}
            if name:
                fact = {"name": name, **fact, "subject": name}
            facts.append(fact)
        shots = [{**f, **({"photographer_note": n} if n else {})} for f, n in zip(facts, notes)]
        # True facts only for a known species (its Wikipedia intro, kept locally);
        # never a lookup online with the photos' subjects or notes.
        for shot, k in zip(shots, known):
            if k["facts"]:
                shot["reference"] = k["facts"]
        # Camera details go into the plan (to decide where they show), and into the
        # writing only for LENS_MOOD.
        cameras = [{**shot, "camera": d} if d else shot for shot, d in zip(shots, details)]
        n = len(images)
        mood = None
        if style is None:
            mood = self._pick_mood(shots, brief, writer, seed, avoid)
            style = MOODS[mood]
        lens = style == MOODS[LENS_MOOD] or bool(shot_list and shot_list.get("camera"))
        if lens:
            shots = cameras  # the hour, lens and shutter speed are this story's material
        common = (
            f"Tone and approach: {style} "
            + (f"Background from the person who took the photos: {brief.strip()} " if brief.strip() else "")
            + ("The viewer sees each picture while the narration plays: point at what is visible only where the "
               "moment's job asks for it (the light, feathers, posture, colour, the background), and say what it "
               "adds rather than list it. " if shot_list else
               "The viewer sees each picture while the narration plays, so never describe what is visible "
               "(colours, poses, perches, grass, sky). Use what is in the pictures only as the starting point, "
               "and give the viewer what they can't see: the story, the stakes, feelings, meaning, jokes, true facts. ")
            + ("Talk about taking the shot, but never say 'photo', 'image', 'picture', 'frame' or 'slide'. " if lens
               else "Never mention a photo, image, picture, camera, frame, shot, slide or background. "
               if not shot_list else "Never say 'photo', 'image', 'picture' or 'slide'. ")
            + "Speak to the viewer, in short sentences that are easy to say aloud. "
            "Real-world facts (numbers, habits, abilities, records) may only come from the 'reference' text given "
            "for a moment; never invent or embellish one, and if there is no reference, use no facts. "
            "Don't label facts ('fun fact'). Use common names, never scientific names. "
            "When a moment has a 'name' (given by the person who took the photos, always correct), call its subject "
            "by that name at least once in that moment's narration, early on (e.g. 'the pintail', 'a northern "
            "wheatear'); never call it just 'the bird', 'the duck' or 'it' without the name. Without a 'name', use "
            "the most specific name in 'subject'. "
        )
        # Plan first, so the moments add up to one story instead of n captions.
        has_camera = ask_show and any(details)
        seen = cameras if has_camera or lens else shots
        jobs = _shot_jobs(shot_list, n) if shot_list else None
        plan_prompt = (
            f"You are writing a narrated slideshow of {n} moments, in {language}. {common}"
            f"What an observer saw in each moment, in order: {json.dumps(seen, ensure_ascii=False)}. "
            + (f"Each moment's job is fixed, in order: {json.dumps(jobs, ensure_ascii=False)}. Plan the narration "
               "as one story in this tone that does those jobs: a title, a one-sentence premise, and the beats "
               "(each moment's job as given). "
               "Also 'motions': for each moment, in order, the camera move over the picture that suits where "
               f"the subject sits in it ('framing'), one of {json.dumps(MOVES)}: zoom in when it is small in a "
               "wide scene, zoom out when it fills the picture, pan towards the side it faces or moves to, so the "
               "move ends on it; not the same move every time. "
               f"Also 'cover': the number (1 to {n}) of the strongest moment for the cover: the sharpest, most "
               "striking subject, eye contact or a peak pose. "
               "Also a 'hook': a thumbnail text overlay of 3 to 4 words for that cover, punchy and in this "
               "tone (no quotes, no hashtags), "
               if jobs else
               f"Plan the narration as one story in this tone: a title, a one-sentence premise, and exactly {n} beats, "
               "one per moment in order, each saying what that moment's narration does for the story "
               "(not what the picture shows). "
               "Also an 'opening': the very first line the viewer hears, at most 12 words, that stops someone "
               "scrolling past: a question, a surprise or a bold promise of what's coming, in this tone, and true "
               "(no invented facts). "
               "Also a 'hook': a thumbnail headline of 2 to 5 words that makes people want to watch (no quotes, no "
               "hashtags), ")
            + f"and an 'outro': one short closing line to be spoken at the very end, under {outro_words} words, in the "
            "same tone, that asks the viewer to follow for more stories like this; it must say 'follow' (in the "
            "story's language), and no names, handles or hashtags. "
            + ("Some moments have 'camera': the settings the photo was taken with. They can be shown as a small "
               "line of text on that moment's slide (never spoken). Say for each moment, in order, whether showing "
               "them fits this story and tone ('show_details': true or false; false when a moment has no "
               "'camera'): yes in a documentary, a nature film or a photographer's memoir, where the craft is "
               "part of the story (then on most moments that have them), no where it would break the spell "
               "(a bedtime story, a joke, a noir mystery). "
               if has_camera else "")
            + ('Answer only with JSON: {"title": "...", "premise": "...", "motions": ["...", ...], "cover": 1, '
               '"hook": "...", "outro": "..."' if jobs else
               'Answer only with JSON: {"title": "...", "premise": "...", "beats": ["...", ...], "opening": "...", '
               '"hook": "...", "outro": "..."') + (', "show_details": [true, ...]' if has_camera else "") + "}"
        )
        properties = {"title": {"type": "string"}, "premise": {"type": "string"}}
        if jobs:
            properties.update(motions={"type": "array", "items": {"type": "string", "enum": MOVES},
                                       "minItems": n, "maxItems": n},
                              cover={"type": "integer", "minimum": 1, "maximum": n})
        else:
            properties.update(beats={"type": "array", "items": {"type": "string"}, "minItems": n, "maxItems": n},
                              opening={"type": "string"})
        properties.update(hook={"type": "string"}, outro={"type": "string"})
        if has_camera:
            properties["show_details"] = {"type": "array", "items": {"type": "boolean"}, "minItems": n,
                                          "maxItems": n}
        plan = _llm_json(writer, plan_prompt, {"type": "object", "properties": properties,
                                                  "required": list(properties)}, temperature, seed)
        beats = jobs or [str(b).strip() for b in plan.get("beats", [])]
        beats += [""] * (n - len(beats))
        # A shot list's first job is the hook: no opening line before it.
        opening = "" if jobs else str(plan.get("opening", "")).strip().strip('"')
        print(f"[documentary] opening: {opening!r}")
        texts, named = [], set()
        for index, (shot, beat) in enumerate(zip(shots, beats)):
            comfy.model_management.throw_exception_if_processing_interrupted()
            prompt = (
                f"You write the narration for moment {index + 1} of {n} of a narrated slideshow titled "
                f"\"{plan.get('title', '')}\". {common}"
                f"The story's premise: {plan.get('premise', '')} "
                f"This moment's job in the story: {beat} "
                + (f"This is the first moment: start its narration with exactly this line, then go on: "
                   f"{json.dumps(opening, ensure_ascii=False)}. " if index == 0 and opening else "")
                + ((f"This moment's subject is the {shot['name']}: say '{shot['name']}' in the narration. "
                    if shot["name"] not in named else
                    f"This moment's subject is the same {shot['name']} the story has already named: don't say "
                    "its name again; call it 'it' or a short word for what it is ('the bird', 'the fox'), and spend the words on "
                    "the story. ") if shot.get("name") else "")
                + (f"Nobody knows this animal's species: call it only {json.dumps(shot.get('subject', ''))} or "
                   "a word like it, never by a species name. " if known[index]["animal"] else "")
                + f"What an observer saw in this moment (the viewer sees it too; don't list it): "
                # Without 'framing': it is for the plan's camera moves, and the writer would say it.
                f"{json.dumps({k: v for k, v in shot.items() if k != 'framing'}, ensure_ascii=False)}. "
                + (f"Write {shot_list['words'][0]} to {words} words, strictly, to be spoken aloud in {language}. "
                   if shot_list else f"Write {words} words at most, to be spoken aloud in {language}. ")
                + "Vary how sentences start, and never use the phrase 'right now'. "
                + (f"The moments so far were narrated like this; continue the story from them without repeating "
                   f"their phrases, images or jokes: {json.dumps(texts, ensure_ascii=False)}. " if texts else "")
                + 'Answer only with JSON: {"narration": "..."}'
            )
            reply = _llm_json(writer, prompt, {"type": "object", "properties": {"narration": {"type": "string"}},
                                                  "required": ["narration"]}, temperature, seed + index + 1)
            text = str(reply.get("narration", "")).strip()
            if not text:
                raise RuntimeError(f"{writer} wrote no narration for image {index + 1}; run again or change the seed")
            # The writer overruns "at most N words", and Narrate can only speed up so much;
            # a shot list's limit is strict (two tries).
            for attempt in range(2 if shot_list else 1):
                if len(text.split()) <= (words if shot_list else words * 1.2):
                    break
                print(f"[documentary] moment {index + 1}: {len(text.split())} words for {words}: shortening")
                short = (f"{prompt} Your first answer was: {json.dumps(text, ensure_ascii=False)}. It has "
                         f"{len(text.split())} words; the limit is {words}. Write it again in at most {words} words, "
                         "keeping its best line.")
                reply = _llm_json(writer, short, {"type": "object", "properties": {"narration": {"type": "string"}},
                                                  "required": ["narration"]}, temperature,
                                  seed + index + 201 + 50 * attempt)
                text = str(reply.get("narration", "")).strip() or text
            # True numbers: the species' facts, the photographer's note (a date, a
            # count) and, for LENS_MOOD, the camera settings.
            known_text = f"{shot.get('reference', '')} {shot.get('photographer_note', '')} {shot.get('camera', '')}"
            if shot_list:
                known_text += " " + " ".join(str(i) for i in range(1, n + 1))  # "your favourite, from 1 to 5"
            invented = _unsupported_numbers(text, known_text)
            if invented:
                # Once more without them; then without the sentences that still have them.
                print(f"[documentary] moment {index + 1}: numbers not in its facts ({', '.join(invented)}): rewriting")
                fix = (f"{prompt} Your first answer was: {json.dumps(text, ensure_ascii=False)}. It states "
                       f"{', '.join(invented)}, which neither the 'reference' nor the photographer's note says: "
                       "write it again without those claims and without any other number they don't have.")
                reply = _llm_json(writer, fix, {"type": "object", "properties": {"narration": {"type": "string"}},
                                                "required": ["narration"]}, temperature, seed + index + 101)
                text = str(reply.get("narration", "")).strip() or text
                invented = _unsupported_numbers(text, known_text)
                if invented:
                    text = _drop_sentences_with(text, invented)
            ask = shot_list.get("ask") if shot_list and index == n - 1 else None
            if ask and not any(w in text.lower() for w in ask):
                # The closing question is the point of the last moment.
                print(f"[documentary] moment {index + 1}: no question to the viewer: rewriting")
                again = (f"{prompt} Your first answer was: {json.dumps(text, ensure_ascii=False)}. It doesn't do "
                         f"the job's question to the viewer: write it again, ending with that question, in at most "
                         f"{words} words.")
                reply = _llm_json(writer, again, {"type": "object", "properties": {"narration": {"type": "string"}},
                                                  "required": ["narration"]}, temperature, seed + index + 301)
                text = str(reply.get("narration", "")).strip() or text
                if not any(w in text.lower() for w in ask):
                    text = f"{text} {shot_list['ask_line'].replace('{n}', str(n))}"
            texts.append(text)
            named.add(shot.get("name"))
        extra = {k: str(plan.get(k, "")).strip().strip('"') for k in ("hook", "outro")}
        if jobs and not 3 <= len(extra["hook"].split()) <= 4:
            # The cover's text overlay: 3 to 4 words.
            for attempt in range(2):
                reply = _llm_json(writer, (
                    f"A thumbnail text overlay for a narrated reel titled \"{plan.get('title', '')}\" in this tone: "
                    f"{style} The first idea was {json.dumps(extra['hook'], ensure_ascii=False)}, which has "
                    f"{len(extra['hook'].split())} words. Write a punchy one of 3 to 4 words (no quotes, no "
                    'hashtags). Answer only with JSON: {"hook": "..."}'),
                    {"type": "object", "properties": {"hook": {"type": "string"}}, "required": ["hook"]},
                    temperature, seed + 401 + attempt)
                hook = str(reply.get("hook", "")).strip().strip('"')
                if 3 <= len(hook.split()) <= 4:
                    extra["hook"] = hook
                    break
            else:
                extra["hook"] = " ".join(extra["hook"].split()[:4])
        # The writer sometimes leaves out the request itself.
        if language.strip().lower() == "english" and "follow" not in extra["outro"].lower():
            extra["outro"] = f"{extra['outro']} Follow for more stories like this.".strip()
        if has_camera:
            shown = [bool(x) for x in plan.get("show_details", [])][:n]
            extra["show_details"] = shown + [False] * (n - len(shown))
            print(f"[documentary] camera details on slides: "
                  f"{[i + 1 for i, s in enumerate(extra['show_details']) if s and details[i]] or 'none'}")
        if jobs:
            moves = [m if m in MOVES else MOVES[i % len(MOVES)] for i, m in enumerate(plan.get("motions") or [])][:n]
            extra["motions"] = moves + [MOVES[i % len(MOVES)] for i in range(len(moves), n)]
            cover = plan.get("cover")
            extra["cover"] = cover if isinstance(cover, int) and 1 <= cover <= n else None
            print(f"[documentary] camera moves: {', '.join(extra['motions'])}; cover: photo {extra['cover']}, "
                  f"{extra['hook']!r}")
        if mood:
            extra["mood"] = mood
        return facts, texts, str(plan.get("title", "")).strip(), extra

    @staticmethod
    def _pick_mood(shots, brief, writer, seed, avoid=()):
        """The mood (a key of MOODS) that fits what the vision model saw, not
        one of `avoid` (e.g. the last nights' moods) while two others are left."""
        moods = {k: v for k, v in MOODS.items() if v and k not in SHOT_LISTS}
        if len([k for k in moods if k not in avoid]) >= 2:
            moods = {k: v for k, v in moods.items() if k not in avoid}
        seen = [{k: s[k] for k in ("subject", "activity", "setting", "others", "photographer_note") if s.get(k)}
                for s in shots]
        prompt = (
            "You choose the tone for a short narrated slideshow of these photos. "
            f"What an observer saw in each photo, in order: {json.dumps(seen, ensure_ascii=False)}. "
            + (f"Background from the person who took the photos: {brief.strip()} " if brief.strip() else "")
            + f"The tones to choose from: {json.dumps(moods, ensure_ascii=False)}. "
            "Pick the one tone that suits these photos best: what they show (wildlife, landscapes, people, "
            "a city, action, quiet moments), their feeling, and what would make the most watchable story; "
            "not always the first one. "
            'Answer only with JSON: {"mood": "<one of the tone names>", "why": "..."}'
        )
        reply = _llm_json(writer, prompt, {"type": "object",
                                              "properties": {"mood": {"type": "string", "enum": list(moods)},
                                                             "why": {"type": "string"}},
                                              "required": ["mood", "why"]}, 0.5, seed)
        mood = reply.get("mood") if reply.get("mood") in moods else next(iter(moods))
        print(f"[documentary] mood: {mood} ({str(reply.get('why', '')).strip()[:200]})")
        return mood


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
                "outro": ("STRING", {"default": "",
                                     "placeholder": "Optional: the narrator's last line. Empty: the story's own follow request; '-': none",
                                     "tooltip": "Spoken over the follow card at the end of the video"}),
            },
            "optional": {
                "my_voice": ("AUDIO", {"tooltip": "Optional: 10-20 s of you reading, from a Load Audio node. The "
                                                  "narration is spoken in that voice (Chatterbox; English works best)"}),
            },
        }

    RETURN_TYPES = (STORYBOARD, "AUDIO")
    RETURN_NAMES = ("storyboard", "narration")
    FUNCTION = "run"
    CATEGORY = "documentary"

    def run(self, *args, **kwargs):
        return _local_first(lambda: self._run(*args, **kwargs))

    def _run(self, storyboard, voice, speed, engine="kokoro", intensity=0.0, outro="", my_voice=None):
        preset = {**VOICE_PRESETS[DEFAULT_MOOD], **VOICE_PRESETS.get(storyboard.get("mood"), {})}
        if voice != AUTO_VOICE:
            preset["voice"] = voice
        if intensity > 0:
            preset["intensity"] = intensity
        line = outro.strip() or storyboard.get("outro") or DEFAULT_OUTRO
        if line == "-":
            line = ""
        n = len(storyboard["texts"])
        texts = storyboard["texts"] + ([line] if line else [])
        _free_gpu()
        device = comfy.model_management.get_torch_device()
        if my_voice is not None:
            # Only Chatterbox can copy a voice. No pitch shift: it should sound like you.
            if engine == "kokoro":
                print("[documentary] my_voice is connected: using Chatterbox instead of Kokoro")
            preset["reference"] = _own_voice(my_voice)
            preset["semitones"] = 0
            # As you speak: a mood's strong expression and low cfg over-acted your
            # voice and mispronounced words; a higher cfg keeps closer to the
            # recording, accent included, and no mood pace. intensity > 0 still sets it.
            if intensity <= 0:
                preset["intensity"] = min(preset["intensity"], MY_VOICE_INTENSITY)
            preset["cfg"] = max(preset["cfg"], MY_VOICE_CFG)
            preset["stretch"] = 1.0
            engine = "chatterbox"
        if engine == "kokoro":
            clips, timings = _kokoro_clips(texts, preset["voice"], preset["kokoro_speed"] * speed, device)
        else:
            clips, timings = _chatterbox_clips(texts, preset, speed, device)
        if _limit(storyboard) > 0:
            clips, timings = _fit_length(storyboard, clips, timings, n)
        full = np.concatenate(clips)
        audio = {"waveform": torch.from_numpy(full)[None, None, :], "sample_rate": KOKORO_RATE}
        return ({**storyboard, "audio": clips[:n], "timings": timings[:n], "sample_rate": KOKORO_RATE,
                 "outro_text": line, "outro_audio": clips[n] if line else None,
                 "outro_timings": timings[n] if line else []}, audio)


def _reel_seconds(photo_seconds, closing_seconds, n):
    """How long a reel will be (Ken Burns Video's reel timing)."""
    closing = max(REEL_OUTRO, closing_seconds + 1.1) if closing_seconds else REEL_OUTRO
    return REEL_INTRO + photo_seconds + n * sum(REEL_PAUSES) + closing


def _video_seconds(photo_seconds, closing_seconds, n):
    """How long a YouTube video will be (Ken Burns Video's default timing)."""
    closing = max(VIDEO_OUTRO, closing_seconds + VIDEO_TAIL) if closing_seconds else VIDEO_OUTRO
    return VIDEO_INTRO + photo_seconds + n * sum(VIDEO_PAUSES) + closing


def _tempo(clip, rate):
    """The clip `rate` times faster (< 1: slower) at the same pitch, with
    ffmpeg's atempo: it keeps a voice natural, where librosa's phase vocoder
    makes it sound hollow and metallic."""
    if abs(rate - 1.0) < 1e-3:
        return clip
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "f32le", "-ar", str(KOKORO_RATE), "-ac", "1",
         "-i", "pipe:0", "-filter:a", f"atempo={rate:.4f}", "-f", "f32le", "pipe:1"],
        input=np.ascontiguousarray(clip, dtype=np.float32).tobytes(), capture_output=True)
    if proc.returncode:
        raise RuntimeError(f"ffmpeg atempo failed: {proc.stderr.decode(errors='replace')[-500:]}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def _fit_length(storyboard, clips, timings, n, most=1.25):
    """Speed the narration up (at most `most` times) when the video would run
    over its limit; the writer was asked for fewer words, but can overrun."""
    reel = _is_reel(storyboard)
    limit = _limit(storyboard) - 0.5
    photos = sum(len(c) for c in clips[:n]) / KOKORO_RATE
    closing = len(clips[n]) / KOKORO_RATE if len(clips) > n else 0.0
    excess = (_reel_seconds if reel else _video_seconds)(photos, closing, n) - limit
    if excess <= 0:
        return clips, timings
    speech = photos + closing
    rate = min(speech / max(speech - excess, 1e-3), most)
    print(f"[documentary] {'reel' if reel else 'video'} {excess:.1f} s too long: narration {rate:.2f}x faster")
    fast, spans = [], []
    for clip, timing in zip(clips, timings):
        out = _tempo(clip, rate)
        scale = len(out) / len(clip)
        fast.append(out)
        spans.append([(t, round(a * scale), round(b * scale)) for t, a, b in timing])
    return fast, spans


def _kokoro_clips(texts, voice, speed, device):
    """One clip per text, and per clip the (text, start, end) of each spoken part in samples."""
    from huggingface_hub import hf_hub_download

    pipeline = _kokoro_pipeline(voice[0], device)
    voice_path = hf_hub_download(KOKORO_REPO, f"voices/{voice}.pt", local_dir=_kokoro_dir())
    clips, timings = [], []
    for text in texts:
        comfy.model_management.throw_exception_if_processing_interrupted()
        parts, spans, pos = [], [], 0
        for r in pipeline(text, voice=voice_path, speed=speed):
            if r.audio is None:
                continue
            part = r.audio.detach().cpu().numpy()
            parts.append(part)
            spans.append((r.graphemes, pos, pos + len(part)))
            pos += len(part)
        if not parts:
            raise RuntimeError(f"Kokoro produced no audio for: {text[:80]}")
        clips.append(np.concatenate(parts).astype(np.float32))
        timings.append(spans)
    return clips, timings


# Chatterbox (Resemble AI, MIT): expressive TTS with an emotion-intensity
# control ("exaggeration"). It takes its voice from a reference clip; we make
# that clip with Kokoro, so the same voice names work for both engines.
_chatterbox = {}
REFERENCE_TEXT = ("The river bends through the valley, past the old stone bridge, where the morning light "
                  "falls softly on the water and the reeds lean in the wind.")


CHATTERBOX_REPO = "ResembleAI/chatterbox"
CHATTERBOX_FILES = ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]


def _chatterbox_model(device):
    import pathlib

    from chatterbox.tts import ChatterboxTTS
    from huggingface_hub import hf_hub_download

    if "model" not in _chatterbox:
        # The files from_pretrained() would fetch, but into models/chatterbox
        # by name: setting HF_HOME here was too late once huggingface_hub had
        # been imported (it reads it then), and the weights weren't found offline.
        cache = os.path.join(folder_paths.models_dir, "chatterbox", "hf", "hub")
        path = [hf_hub_download(CHATTERBOX_REPO, f, cache_dir=cache) for f in CHATTERBOX_FILES][-1]
        _chatterbox["model"] = ChatterboxTTS.from_local(pathlib.Path(path).parent, device=str(device))
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
        clip = np.concatenate(_kokoro_clips([REFERENCE_TEXT], voice, 0.9, device)[0])
        _write_wav(path, clip, KOKORO_RATE)
    return path


def _own_voice(audio, seconds=20.0):
    """A voice recording (ComfyUI AUDIO) as a mono WAV for Chatterbox: at most
    `seconds` long, silence at the ends trimmed, level normalised."""
    import librosa

    rate = int(audio["sample_rate"])
    wav = audio["waveform"][0].float().mean(0).cpu().numpy()
    wav, _ = librosa.effects.trim(wav, top_db=35)
    if len(wav) < rate * 5:
        raise ValueError(f"my_voice is {len(wav) / rate:.1f} s long; record 10-20 s of speech")
    wav = wav[:int(rate * seconds)]
    wav = wav / (float(np.abs(wav).max()) or 1.0) * 0.9
    path = os.path.join(folder_paths.get_temp_directory(), "documentary_my_voice.wav")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _write_wav(path, wav, rate)
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
        reference = preset.get("reference") or _reference_clip(preset["voice"], device)
        stretch = preset["stretch"] * speed
        clips, timings = [], []
        for text in texts:
            parts, chunks = [], _sentences(text)
            for chunk in chunks:
                comfy.model_management.throw_exception_if_processing_interrupted()
                wav = model.generate(chunk, audio_prompt_path=reference, exaggeration=preset["intensity"],
                                     cfg_weight=preset["cfg"]).squeeze(0).cpu().numpy().astype(np.float32)
                parts.append(wav)
            clip = np.concatenate(parts)
            raw = len(clip)
            if model.sr != KOKORO_RATE:
                clip = librosa.resample(clip, orig_sr=model.sr, target_sr=KOKORO_RATE)
            if preset["semitones"]:
                clip = librosa.effects.pitch_shift(clip, sr=KOKORO_RATE, n_steps=preset["semitones"])
            clip = _tempo(clip, stretch)
            peak = float(np.abs(clip).max()) or 1.0
            clips.append((clip / peak * 0.9).astype(np.float32))
            # Resampling and stretching keep the parts' proportions.
            scale, spans, pos = len(clip) / raw, [], 0
            for chunk, part in zip(chunks, parts):
                spans.append((chunk, round(pos * scale), round((pos + len(part)) * scale)))
                pos += len(part)
            timings.append(spans)
        return clips, timings
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


def _framed(image, width, height, zoom, device, centre=0.5):
    """For a tall frame: a landscape photo as a square over a blurred, darker
    copy of itself, so nothing important is cropped away; a portrait photo
    fills the frame. centre: where the square sits (share of the height)."""
    _, h, w, _ = image.shape
    if w / h < 0.8:
        return _cover(image, width, height, zoom, device)
    out = _blur(_cover(image, width, height, zoom, device)) * 0.5
    square = _cover(image, width, width, zoom, device)
    side, full = square.shape[2], out.shape[2]
    y0 = min(max(round(full * centre - side / 2), 0), full - side)
    out[:, :, y0:y0 + side, :] = square
    return out


def _fitted(image, width, height, zoom, device):
    """For a wide frame: a portrait or square photo whole, at full height over
    a blurred, darker copy of itself (cropping it to 16:9 would keep only a
    band across the middle); a landscape photo fills the frame."""
    _, h, w, _ = image.shape
    if w / h >= 1.2:
        return _cover(image, width, height, zoom, device)
    out = _blur(_cover(image, width, height, zoom, device)) * 0.5
    full_h, full_w = out.shape[2], out.shape[3]
    fit_w = min(round(full_h * w / h), full_w)
    img = image[0].permute(2, 0, 1)[None].to(device, torch.float32)
    photo = F.interpolate(img, size=(full_h, fit_w), mode="bicubic", antialias=True, align_corners=False).clamp(0, 1)
    x0 = (full_w - fit_w) // 2
    out[:, :, :, x0:x0 + fit_w] = photo
    return out


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
    """samples: [n] (mono) or [channels, n]."""
    samples = np.atleast_2d(samples)
    pcm = (np.clip(samples.T, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(samples.shape[0])
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


# Background music with ACE-Step 1.5 (ComfyUI's own nodes, called in code so
# the length follows the narration). Settings as in ComfyUI's ACE-Step 1.5
# template: AuraFlow shift 3, 8 steps, cfg 1, euler/simple.
ACE_CHECKPOINT = "ace_step_1.5_turbo_aio.safetensors"
ACE_REPO = "Comfy-Org/ace_step_1.5_ComfyUI_files"
# Per mood: tags (ACE-Step's style prompt), bpm, key.
MUSIC = {
    "documentary (Nat Geo)": ("cinematic orchestral nature documentary score, warm strings, gentle woodwinds, soft piano, "
                              "harp, curious and full of wonder, light and airy, instrumental", 90, "D major"),
    "funny": ("playful quirky comedy score, pizzicato strings, bassoon, clarinet, ukulele, glockenspiel, light "
              "percussion, bouncy and cheeky, instrumental", 112, "F major"),
    "poetic": ("ambient contemplative score, slow solo piano, soft strings, airy pads, gentle and reflective, "
               "instrumental", 66, "A minor"),
    "bedtime story": ("soft lullaby, music box, celesta, harp, warm pads, very gentle, slow and sleepy, instrumental",
                      60, "C major"),
    "epic trailer": ("epic cinematic trailer music, full orchestra, taiko drums, deep braams, choir, heroic brass, "
                     "rising tension building to a huge climax, instrumental", 100, "D minor"),
    "nostalgic memoir": ("nostalgic warm acoustic guitar and piano, soft strings, bittersweet and gentle, "
                         "instrumental", 80, "G major"),
    "sports commentary": ("energetic sports broadcast theme, driving drums, brass stabs, electric guitar, stadium "
                          "anthem, upbeat and exciting, instrumental", 128, "E major"),
    "mystery / noir": ("film noir jazz, smoky muted trumpet, upright bass, brushed drums, slow piano, moody and "
                       "mysterious, instrumental", 72, "C minor"),
    "mini-documentary": ("atmospheric nature documentary score, soft strings, warm pads, gentle piano, calm and "
                         "observant, instrumental", 72, "D major"),
    "photographer's craft": ("warm acoustic guitar and soft piano, light percussion, thoughtful and appreciative, "
                             "instrumental", 88, "G major"),
    "species spotlight": ("rhythmic modern chill beat, plucked synths, soft marimba, light drums, bright and "
                          "engaging, instrumental", 104, "A major"),
}


def _narration_seconds(storyboard, pause=1.6, tail=3.0, intro=4.5, outro=5.0):
    """About how long the video will be (intro, narration + Ken Burns' default
    pauses, follow card); Ken Burns Video trims or loops the music to the exact length."""
    rate = storyboard["sample_rate"]
    closing = storyboard.get("outro_audio")
    if _is_reel(storyboard):
        return _reel_seconds(sum(len(c) for c in storyboard["audio"]) / rate,
                             len(closing) / rate if closing is not None else 0.0, len(storyboard["audio"])) + 1.0
    closing = len(closing) / rate + 1.5 if closing is not None else 0.0
    return intro + sum(len(c) / rate + pause for c in storyboard["audio"]) + max(outro, closing) + tail


class DocumentaryMusic:
    @classmethod
    def INPUT_TYPES(cls):
        checkpoints = folder_paths.get_filename_list("checkpoints")
        return {
            "required": {
                "storyboard": (STORYBOARD,),
                "checkpoint": (checkpoints or [ACE_CHECKPOINT], {"default": ACE_CHECKPOINT,
                                                                 "tooltip": f"ACE-Step 1.5 checkpoint (from {ACE_REPO})"}),
                "style": ("STRING", {"multiline": True, "default": "",
                                     "placeholder": "Optional: describe the music (genre, instruments, feel). "
                                                    "Empty: chosen from the mood"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 60, "tooltip": "8 for the turbo model"}),
            },
        }

    RETURN_TYPES = (STORYBOARD, "AUDIO")
    RETURN_NAMES = ("storyboard", "music")
    FUNCTION = "run"
    CATEGORY = "documentary"

    def run(self, storyboard, checkpoint, style, seed, steps):
        from comfy_extras.nodes_ace import EmptyAceStep15LatentAudio, TextEncodeAceStepAudio15
        from comfy_extras.nodes_audio import VAEDecodeAudio
        from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow

        if "audio" not in storyboard:
            raise ValueError("Connect the storyboard from 'Narrate Storyboard' (the music follows the narration's length)")
        tags, bpm, key = MUSIC.get(storyboard.get("mood"), MUSIC[DEFAULT_MOOD])
        if style.strip():
            tags = style.strip()
        elif storyboard.get("custom_style"):
            tags = f"instrumental background score for a narration in this style: {storyboard['custom_style'][:300]}"
        seconds = round(_narration_seconds(storyboard), 1)
        _free_gpu()
        model, clip, vae = nodes.CheckpointLoaderSimple().load_checkpoint(checkpoint)[:3]
        positive = TextEncodeAceStepAudio15.execute(clip, tags, "[Instrumental]", seed, bpm, seconds, "4", "en", key,
                                                    True, 2.0, 0.85, 0.9, 0, 0.0)[0]
        negative = nodes.ConditioningZeroOut().zero_out(positive)[0]
        model = ModelSamplingAuraFlow().patch_aura(model, 3.0)[0]
        latent = EmptyAceStep15LatentAudio.execute(seconds, 1)[0]
        samples = nodes.common_ksampler(model, seed, steps, 1.0, "euler", "simple", positive, negative, latent)[0]
        audio = VAEDecodeAudio.execute(vae, samples)[0]
        del model, clip, vae, positive, negative, samples
        # Free GPU 0 for Ken Burns.
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        wave_ = audio["waveform"][0].float().cpu().numpy()
        return ({**storyboard, "music": wave_, "music_rate": audio["sample_rate"]}, audio)


def _resample(x, src, dst):
    if src == dst:
        return x
    import librosa
    return librosa.resample(x, orig_sr=src, target_sr=dst, axis=-1).astype(np.float32)


def _fit_music(music, length, rate, crossfade=2.0):
    """Music ([channels, n]) looped with a crossfade, or trimmed, to length."""
    out = music
    cf = min(int(crossfade * rate), music.shape[1] // 4)
    while out.shape[1] < length:
        ramp = np.linspace(0, 1, cf, dtype=np.float32)
        head = out[:, :-cf] if cf else out
        joint = out[:, -cf:] * (1 - ramp) + music[:, :cf] * ramp if cf else np.zeros((music.shape[0], 0), np.float32)
        out = np.concatenate([head, joint, music[:, cf:]], axis=1)
    return out[:, :length]


def _mix(narration, rate, speech, music, music_rate, volume, duck):
    """Narration (mono) over music: music normalised, at `volume`, dipping to
    volume * duck while the narrator speaks (speech: 0/1 per sample), faded in
    and out. Returns [2, n] at the music's rate."""
    out_rate = music_rate
    voice = _resample(narration, rate, out_rate)
    n = voice.shape[0]
    mask = _resample(speech.astype(np.float32), rate, out_rate)[:n]
    mask = np.pad(mask, (0, n - mask.shape[0]))
    # Smooth the ducking over ~0.4 s, so the music doesn't jump.
    width = int(0.4 * out_rate)
    kernel = np.ones(width, dtype=np.float32) / width
    mask = np.clip(np.convolve(mask, kernel, mode="same") * 1.5, 0, 1)
    if music.shape[0] == 1:
        music = np.repeat(music, 2, axis=0)
    music = _fit_music(music[:2], n, out_rate)
    rms = float(np.sqrt(np.mean(music ** 2))) or 1.0
    music = music / rms * 0.12
    gain = volume * (1 - (1 - duck) * mask)
    fade_in, fade_out = min(int(1.0 * out_rate), n), min(int(2.5 * out_rate), n)
    gain[:fade_in] *= np.linspace(0, 1, fade_in, dtype=np.float32)
    gain[n - fade_out:] *= np.linspace(1, 0, fade_out, dtype=np.float32)
    mixed = voice[None, :] + music * gain[None, :]
    peak = float(np.abs(mixed).max())
    if peak > 0.98:
        mixed *= 0.98 / peak
    return mixed.astype(np.float32), out_rate


# Text and graphics for the intro, subtitles, follow card and thumbnail:
# drawn once with PIL as RGBA layers ([4,h,w], straight alpha, on the GPU),
# then moved, scaled and blended per frame.
FONT_DIR = "/usr/share/fonts/truetype/dejavu"
FONTS = {"sans": "DejaVuSans-Bold.ttf", "serif": "DejaVuSerif-Bold.ttf", "regular": "DejaVuSans.ttf"}
ACCENT = (255, 200, 61)


def _font(kind, size):
    try:
        return ImageFont.truetype(os.path.join(FONT_DIR, FONTS[kind]), max(int(size), 8))
    except OSError:
        return ImageFont.load_default(max(int(size), 8))


def _wrap(text, font, max_width):
    lines, line = [], ""
    for word in text.split():
        test = f"{line} {word}".strip()
        if line and font.getlength(test) > max_width:
            lines.append(line)
            line = word
        else:
            line = test
    return lines + ([line] if line else [])


def _balanced(text, font, max_width):
    """Like _wrap, but with lines of similar length (no lone last word)."""
    lines = _wrap(text, font, max_width)
    width = max_width
    while 1 < len(lines) < len(text.split()):  # every word on its own line: can't get narrower
        narrower = _wrap(text, font, width * 0.95)
        if len(narrower) > len(lines):
            break
        lines, width = narrower, width * 0.95
    return lines


def _fit(text, kind, size, max_width, max_lines, min_size=10):
    """Font and lines: the largest size up to `size` at which text wraps into max_lines."""
    while True:
        font = _font(kind, size)
        lines = _balanced(text, font, max_width)
        if len(lines) <= max_lines or size <= min_size:
            return font, lines
        size = int(size * 0.92)


def _layer(img, device):
    return torch.from_numpy(np.array(img.convert("RGBA"))).to(device).permute(2, 0, 1).float() / 255


def _text_layer(lines, font, device, fill=(255, 255, 255), stroke=0, shadow=0, align="center", spacing=0.12,
                tracking=0):
    """Lines of text as an RGBA layer, with an optional black outline and a soft drop shadow.
    tracking: extra pixels between letters (for small capitals)."""
    ascent, descent = font.getmetrics()
    line_h = ascent + descent
    gap = round(line_h * spacing)

    def length(line):
        return font.getlength(line) + tracking * max(len(line) - 1, 0)

    widths = [length(line) for line in lines]
    pad = stroke + 3 * shadow + 2
    w = math.ceil(max(widths, default=1)) + 2 * pad
    h = line_h * len(lines) + gap * max(len(lines) - 1, 0) + 2 * pad

    def draw(img, color, offset):
        d = ImageDraw.Draw(img)
        y = pad + offset
        for line, lw in zip(lines, widths):
            x = pad + offset + ((w - 2 * pad - lw) / 2 if align == "center" else 0)
            if tracking:
                for ch in line:
                    d.text((x, y), ch, font=font, fill=color, stroke_width=stroke, stroke_fill=(0, 0, 0, color[3]))
                    x += font.getlength(ch) + tracking
            else:
                d.text((x, y), line, font=font, fill=color, stroke_width=stroke, stroke_fill=(0, 0, 0, color[3]))
            y += line_h + gap

    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    if shadow:
        draw(img, (0, 0, 0, 180), shadow)
        img = img.filter(ImageFilter.GaussianBlur(shadow))
    draw(img, (*fill, 255), 0)
    return _layer(img, device)


def _pill(text, font, device, colors=((245, 133, 41), (221, 42, 123), (129, 52, 175)), alpha=255):
    """A rounded button with a left-to-right gradient (default: Instagram's) and centred white text."""
    ascent, descent = font.getmetrics()
    w, h = round(font.getlength(text) + 2.2 * ascent), round((ascent + descent) * 1.9)
    stops = np.array(colors, dtype=np.float32)
    pos = np.linspace(0, len(stops) - 1, w)
    k = np.minimum(pos.astype(int), len(stops) - 2)
    row = stops[k] + (stops[k + 1] - stops[k]) * (pos - k)[:, None]
    img = Image.fromarray(np.repeat(row[None], h, axis=0).astype(np.uint8)).convert("RGBA")
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, w - 1, h - 1), radius=h // 2, fill=alpha)
    img.putalpha(mask)
    ImageDraw.Draw(img).text((w / 2, h / 2), text, font=font, fill=(255, 255, 255, 255), anchor="mm")
    return _layer(img, device)


def _card(image, side, angle, device):
    """The photo as a print with a white border, turned by `angle` degrees, with a soft shadow."""
    arr = (image[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    pil = Image.fromarray(arr).convert("RGB")
    pil.thumbnail((side, side), Image.LANCZOS)
    b = max(3, round(side * 0.03))
    card = Image.new("RGBA", (pil.width + 2 * b, pil.height + 2 * b), (250, 248, 242, 255))
    card.paste(pil, (b, b))
    pad = max(6, round(side * 0.07))
    canvas = Image.new("RGBA", (card.width + 2 * pad, card.height + 2 * pad), (0, 0, 0, 0))
    canvas.paste((0, 0, 0, 150), (pad + b, pad + 2 * b, pad + b + card.width, pad + 2 * b + card.height))
    canvas = canvas.filter(ImageFilter.GaussianBlur(pad / 2.5))
    canvas.alpha_composite(card, (pad, pad))
    return _layer(canvas.rotate(angle, resample=Image.BICUBIC, expand=True), device)


def _scaled(layer, s):
    if abs(s - 1) < 1e-3:
        return layer
    _, h, w = layer.shape
    size = (max(round(h * s), 1), max(round(w * s), 1))
    return F.interpolate(layer[None], size=size, mode="bilinear", antialias=s < 1, align_corners=False)[0]


def _paste(frame, layer, cx, cy, opacity=1.0):
    """Blend an RGBA layer into frame ([1,3,H,W], changed in place), centred at (cx, cy)."""
    if opacity <= 0:
        return frame
    _, _, H, W = frame.shape
    _, h, w = layer.shape
    x, y = round(cx - w / 2), round(cy - h / 2)
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1:
        return frame
    part = layer[:, y0 - y:y1 - y, x0 - x:x1 - x]
    a = part[3:4] * min(opacity, 1.0)
    region = frame[0, :, y0:y1, x0:x1]
    frame[0, :, y0:y1, x0:x1] = region * (1 - a) + part[:3] * a
    return frame


def _blur(img, factor=20):
    _, _, h, w = img.shape
    small = F.interpolate(img, size=(max(h // factor, 2), max(w // factor, 2)), mode="area")
    small = F.avg_pool2d(F.pad(small, (2, 2, 2, 2), mode="replicate"), 5, stride=1)
    return F.interpolate(small, size=(h, w), mode="bicubic", align_corners=False).clamp(0, 1)


def _vignette(width, height, device, strength=0.55):
    y = torch.linspace(-1, 1, height, device=device)[:, None]
    x = torch.linspace(-1, 1, width, device=device)[None, :]
    return (1 - strength * (x ** 2 + y ** 2) / 2).clamp(0, 1)[None, None]


def _ease_out_back(t, c=1.70158):
    return 1 + (c + 1) * (t - 1) ** 3 + c * (t - 1) ** 2


def _ease_cubic(t):
    return 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def _ease_quint(t):
    return 16 * t ** 5 if t < 0.5 else 1 - (-2 * t + 2) ** 5 / 2


def _ramp(t, start, length):
    return min(max((t - start) / length, 0.0), 1.0) if length > 0 else float(t >= start)


def _zoom(img, s):
    theta = torch.tensor([[1 / s, 0, 0], [0, 1 / s, 0]], dtype=img.dtype, device=img.device)[None]
    grid = F.affine_grid(theta, img.shape, align_corners=False)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=False)


# Transitions between scenes: f(a, b, p) with p from 0 (all a) to 1 (all b).
TRANSITIONS = ["dissolve", "push left", "push up", "whip pan", "zoom through", "iris", "diagonal wipe",
               "light leak", "flash", "fade to black"]
# What "mix" picks from (no repeats in a row); the intro always zooms through into photo 1.
MIX = ["whip pan", "zoom through", "iris", "light leak", "diagonal wipe", "flash", "push left", "dissolve", "push up"]


class _Transitions:
    def __init__(self, width, height, device):
        y = torch.linspace(-1, 1, height, device=device)[:, None].expand(height, width)
        x = torch.linspace(-1, 1, width, device=device)[None, :].expand(height, width)
        self.aspect = width / height
        self.x, self.y = x, y
        # 0 in the centre, 1 in the corners; and 0 top left -> 1 bottom right.
        self.r = torch.sqrt((x * self.aspect) ** 2 + y ** 2) / math.sqrt(self.aspect ** 2 + 1)
        self.d = (x + 1) / 2 * 0.6 + (y + 1) / 2 * 0.4

    @staticmethod
    def _push(a, b, p, dim):
        size = a.shape[dim]
        off = min(max(round(p * size), 0), size)
        return torch.cat([a.narrow(dim, off, size - off), b.narrow(dim, 0, off)], dim=dim)

    def __call__(self, kind, a, b, p):
        if kind == "dissolve":
            e = _ease_cubic(p)
            return a + (b - a) * e
        if kind == "fade to black":
            return a * max(1 - 2 * p, 0) if p < 0.5 else b * min(2 * p - 1, 1)
        if kind in ("push left", "push up"):
            return self._push(a, b, _ease_cubic(p), 3 if kind == "push left" else 2)
        if kind == "whip pan":
            # A fast push with motion blur along the move.
            e, blur = _ease_quint(p), 0.14 * math.sin(math.pi * p)
            taps = [e + blur * (k / 6 - 0.5) for k in range(7)] if blur > 0.004 else [e]
            return sum(self._push(a, b, min(max(t, 0), 1), 3) for t in taps) / len(taps)
        if kind == "zoom through":
            e, blur = _ease_cubic(p), math.sin(math.pi * p)
            m = min(max((p - 0.35) / 0.3, 0), 1)
            sa, sb = 1 + 1.2 * e, 1 + 0.6 * (1 - e)
            za = sum(_zoom(a, sa * (1 + 0.035 * k * blur)) for k in range(4)) / 4 if m < 1 else 0
            zb = sum(_zoom(b, sb * (1 + 0.035 * k * blur)) for k in range(4)) / 4 if m > 0 else 0
            out = za * (1 - m) + zb * m
            flash = 0.3 * blur ** 2
            return out + (1 - out) * flash
        if kind == "iris":
            e, soft = _ease_cubic(p), 0.05
            edge = e * (1 + 2 * soft)
            m = ((edge - self.r) / soft).clamp(0, 1)[None, None]
            ring = (torch.exp(-((self.r - edge + soft / 2) / 0.012) ** 2) * 0.6 * math.sin(math.pi * p))[None, None]
            out = a * (1 - m) + b * m
            return out + (1 - out) * ring
        if kind == "diagonal wipe":
            e, soft = _ease_cubic(p), 0.12
            edge = e * (1 + soft)
            m = ((edge - self.d) / soft).clamp(0, 1)[None, None]
            glow = (torch.exp(-((self.d - edge + soft / 2) / 0.03) ** 2) * 0.45 * math.sin(math.pi * p))[None, None]
            out = a * (1 - m) + b * m
            return out + (1 - out) * glow
        if kind == "light leak":
            # A warm glow sweeps across while the pictures cross; screen blend.
            m = min(max((p - 0.2) / 0.6, 0), 1)
            out = a + (b - a) * _ease_cubic(m)
            strength = math.sin(math.pi * p) * 1.1
            glow = 0
            for cx, cy, size, color in ((-1.2 + 2.6 * p, -0.4 + 0.5 * p, 0.55, (1.0, 0.55, 0.22)),
                                        (-0.4 + 2.0 * p, 0.5 - 0.6 * p, 0.35, (1.0, 0.75, 0.35))):
                blob = torch.exp(-(((self.x - cx) * self.aspect) ** 2 + (self.y - cy) ** 2) / size)
                glow = glow + blob[None, None] * torch.tensor(color, device=a.device)[None, :, None, None]
            glow = (glow * strength).clamp(0, 1)
            return 1 - (1 - out) * (1 - glow)
        if kind == "flash":
            white = math.sin(math.pi * p) ** 2
            base = a if p < 0.5 else b
            return base + (1 - base) * white
        return b if p >= 0.5 else a  # cut


class _PhotoScene:
    """One photo, panned and zoomed over the time it is on screen."""

    def __init__(self, image, index, motion, zoom, rng, width, height, device, tall=False, details=""):
        self.image, self.zoom, self.size, self.device, self.tall = image, zoom, (width, height), device, tall
        self.moves = _motion(index, motion, zoom, rng)
        self.base = None
        self.details, self.details_layer = details, None

    def frame(self, u, t):
        width, height = self.size
        if self.base is None:
            if self.tall:
                # A little above the middle, so the subtitles fit under the photo.
                self.base = _framed(self.image, width, height, self.zoom, self.device, centre=0.44)
            else:
                self.base = _fitted(self.image, width, height, self.zoom, self.device)
            if self.details:
                unit = min(width, height)
                font, lines = _fit(self.details, "regular", unit * (0.03 if self.tall else 0.024),
                                   width * (0.86 if self.tall else 0.6), 2)
                self.details_layer = _text_layer(lines, font, self.device, fill=(235, 235, 235),
                                                 shadow=max(2, round(unit * 0.003)),
                                                 align="center" if self.tall else "left")
        # Partly linear, so the picture still moves during the transitions.
        u = 0.55 * u + 0.45 * _ease(u)
        (s0, x0, y0), (s1, x1, y1) = self.moves
        frame = _frame(self.base, s0 + (s1 - s0) * u, x0 + (x1 - x0) * u, y0 + (y1 - y0) * u, width, height)
        if self.details_layer is not None:
            # The camera details, small: top left in a video, above the photo in a
            # reel (under Instagram's top bar); they fade in after the cut.
            layer = self.details_layer
            if self.tall:
                cx, cy = width / 2, height * 0.115
            else:
                margin = min(width, height) * 0.04
                cx, cy = margin + layer.shape[2] / 2, margin + layer.shape[1] / 2
            frame = _paste(frame.contiguous(), layer, cx, cy, 0.9 * _ramp(t, 0.5, 0.6))
        return frame

    def release(self):
        self.base = None
        self.details_layer = None


class _IntroScene:
    """The photos drop in one after another as tilted prints over a blurred
    background, then the title and the mood's label appear."""

    def __init__(self, images, title, label, width, height, seconds, rng, device):
        self.size = (width, height)
        self.bg = _blur(_cover(images[0], width, height, 1.1, device)) * 0.42 * _vignette(
            round(width * 1.1), round(height * 1.1), device)
        n = len(images)
        unit = min(width, height)
        cols = max(1, round(math.sqrt(n * width / height * 0.9)))
        rows = math.ceil(n / cols)
        cw, ch = width * 0.9 / cols, height * 0.84 / rows
        side = round(min(cw, ch) * 1.15)
        self.cards = []
        stagger = min(0.24, seconds * 0.38 / n)
        order = list(range(n))
        rng.shuffle(order)
        for k in range(n):
            r, c = divmod(k, cols)
            in_row = min(cols, n - r * cols)
            cx = width / 2 + (c - (in_row - 1) / 2) * cw + rng.uniform(-0.1, 0.1) * cw
            cy = height / 2 + (r - (rows - 1) / 2) * ch + rng.uniform(-0.1, 0.1) * ch
            layer = _card(images[k], side, rng.uniform(-9, 9), device)
            self.cards.append((0.15 + order[k] * stagger, cx, cy, layer))
        self.cards.sort(key=lambda c: c[0])  # later cards on top
        self.title_at = min(self.cards[-1][0] + 0.55, seconds * 0.5)
        font, lines = _fit(title or "", "serif", unit * 0.095, width * 0.8, 3 if height > width else 2)
        self.title = _text_layer(lines, font, device, shadow=max(2, round(unit * 0.006))) if lines else None
        small = _font("sans", unit * 0.03)
        self.label = _text_layer([label.upper()], small, device, fill=ACCENT,
                                 tracking=round(unit * 0.006)) if label else None
        # A dark band across the middle, so the title reads over the photos.
        band_h = (self.title.shape[1] if self.title is not None else 0) + (
            self.label.shape[1] * 1.6 if self.label is not None else 0) + unit * 0.08
        yy = torch.linspace(-1, 1, round(band_h), device=device)[:, None]
        alpha = (0.62 * (1 - yy.abs() ** 4)).expand(-1, width)
        self.band = torch.cat([torch.zeros(3, *alpha.shape, device=device), alpha[None]], 0)

    def frame(self, u, t):
        width, height = self.size
        g = 1 + 0.06 * u  # slow push in on everything
        frame = _frame(self.bg, 1 + 0.06 * u, 0, 0, width, height).contiguous()
        for start, cx, cy, layer in self.cards:
            a = _ramp(t, start, 0.55)
            if a <= 0:
                continue
            s = (1.4 - 0.4 * _ease_out_back(a)) * g
            _paste(frame, _scaled(layer, s), width / 2 + (cx - width / 2) * g, height / 2 + (cy - height / 2) * g,
                   min(a * 2.5, 1))
        a = _ramp(t, self.title_at, 0.6)
        if a > 0:
            e = _ease(a)
            _paste(frame, self.band, width / 2, height / 2, e)
            label_h = self.label.shape[1] if self.label is not None else 0
            if self.title is not None:
                _paste(frame, _scaled(self.title, 1.08 - 0.08 * e), width / 2, height / 2 - label_h * 0.55, e)
                ty = height / 2 + self.title.shape[1] / 2 + label_h * 0.15
            else:
                ty = height / 2
            if self.label is not None:
                _paste(frame, self.label, width / 2, ty, _ramp(t, self.title_at + 0.3, 0.5))
        return frame

    def release(self):
        pass


class _OutroScene:
    """The follow card: the last photo blurred, the title, a follow request,
    the handle, a pulsing Follow button and like / share / save."""

    def __init__(self, image, title, follow, handle, width, height, seconds, device):
        self.size, self.seconds = (width, height), seconds
        self.bg = _blur(_cover(image, width, height, 1.1, device)) * 0.36 * _vignette(
            round(width * 1.1), round(height * 1.1), device)
        items = []  # (layer, gap above as a share of height, pulse)
        unit = min(width, height)
        if title:
            font, lines = _fit(title, "serif", unit * 0.045, width * 0.8, 2)
            items.append((_text_layer(lines, font, device, fill=(225, 225, 225), shadow=2), 0, False))
        if follow:
            font, lines = _fit(follow, "sans", unit * 0.08, width * 0.82, 3)
            items.append((_text_layer(lines, font, device, shadow=max(2, round(unit * 0.005))), 0.035, False))
        if handle:
            handle = handle if handle.startswith("@") else f"@{handle}"
            items.append((_text_layer([handle], _font("sans", unit * 0.058), device, fill=ACCENT, shadow=2),
                          0.012, False))
        items.append((_pill("+  Follow", _font("sans", unit * 0.045), device), 0.045, True))
        items.append((_text_layer(["♥ Like   ·   Share   ·   Save"], _font("regular", unit * 0.036),
                                  device, fill=(235, 235, 235), shadow=2), 0.04, False))
        total = sum(layer.shape[1] + gap * height for layer, gap, _ in items)
        y = height * 0.45 - total / 2
        self.items = []
        for k, (layer, gap, pulse) in enumerate(items):
            y += gap * height
            self.items.append((layer, y + layer.shape[1] / 2, 0.2 + 0.22 * k, pulse))
            y += layer.shape[1]

    def frame(self, u, t):
        width, height = self.size
        frame = _frame(self.bg, 1 + 0.05 * u, 0, 0, width, height).contiguous()
        for layer, cy, start, pulse in self.items:
            a = _ramp(t, start, 0.5)
            if a <= 0:
                continue
            e = _ease(a)
            if pulse and t > start + 0.5:
                layer = _scaled(layer, 1 + 0.045 * math.sin(2 * math.pi * 0.9 * (t - start - 0.5)))
            _paste(frame, layer, width / 2, cy + (1 - e) * height * 0.04, e)
        end = self.seconds - t
        if end < 0.8:
            frame = frame * max(end / 0.8, 0)
        return frame

    def release(self):
        pass


def _subtitle_cues(text, start, end, font, max_width, max_lines=2):
    """On-screen captions of at most max_lines lines for text spoken from start
    to end (seconds), timed in proportion to their length."""
    import re

    pieces = []
    for sentence in re.split(r"(?<=[.!?…])\s+", text.strip()):
        lines = _balanced(sentence, font, max_width)
        # Split long sentences into balanced pieces, not 2 + 1 lines.
        count = math.ceil(len(lines) / max_lines)
        if count > 1:
            words, per = sentence.split(), math.ceil(len(sentence.split()) / count)
            for k in range(0, len(words), per):
                pieces.append(_balanced(" ".join(words[k:k + per]), font, max_width)[:max_lines + 1])
        elif lines:
            pieces.append(lines)
    total = sum(len(" ".join(p)) for p in pieces) or 1
    cues, t = [], start
    for p in pieces:
        d = (end - start) * len(" ".join(p)) / total
        cues.append((t, t + d, p))
        t += d
    return cues


class KenBurnsVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "storyboard": (STORYBOARD,),
                "width": ("INT", {"default": 1920, "min": 256, "max": 3840, "step": 2,
                                  "tooltip": "For a YouTube video; a reel is always 1080x1920"}),
                "height": ("INT", {"default": 1080, "min": 256, "max": 2160, "step": 2,
                                   "tooltip": "For a YouTube video; a reel is always 1080x1920"}),
                "fps": ("INT", {"default": 30, "min": 10, "max": 60}),
                "zoom": ("FLOAT", {"default": 1.2, "min": 1.0, "max": 2.0, "step": 0.05}),
                "motion": (["alternate", "random"] + MOVES + [STORY_MOTION],
                           {"tooltip": "The camera move over each photo; 'auto (from story)': the one the story "
                                       "chose for it (a reel format such as 'mini-documentary'), else alternate"}),
                "pause_before": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 5.0, "step": 0.1,
                                           "tooltip": "Seconds of image before each narration starts"}),
                "pause_after": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1,
                                          "tooltip": "Seconds of image after each narration ends"}),
                "fade": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.1,
                                   "tooltip": "Length of each transition (seconds); 0 = hard cuts"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                # New inputs go last, so saved workflows keep their widget values.
                "music_volume": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 2.0, "step": 0.05,
                                           "tooltip": "Background music level between narrations (needs Documentary "
                                                      "Music in front; 0 = no music)"}),
                "music_duck": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                                         "tooltip": "Music level while the narrator speaks, as a share of music_volume"}),
                "transition": (["mix"] + TRANSITIONS, {"default": "mix",
                                                       "tooltip": "mix: a different one each time"}),
                "intro_seconds": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 15.0, "step": 0.5,
                                            "tooltip": "Opening collage of all photos with the title; 0 = none"}),
                "outro_seconds": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 15.0, "step": 0.5,
                                            "tooltip": "Follow card at the end (at least as long as the narrator's "
                                                       "closing line); 0 = none"}),
                "follow_text": ("STRING", {"default": "Follow for more stories like this"}),
                "handle": ("STRING", {"default": "", "placeholder": "Optional: your @handle, shown on the follow card"}),
                "subtitles": ("BOOLEAN", {"default": True}),
                "subtitle_size": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "run"
    CATEGORY = "documentary"

    def run(self, storyboard, width, height, fps, zoom, motion, pause_before, pause_after, fade, seed,
            music_volume=0.4, music_duck=0.3, transition="mix", intro_seconds=4.5, outro_seconds=5.0,
            follow_text="Follow for more stories like this", handle="", subtitles=True, subtitle_size=1.0):
        if "audio" not in storyboard:
            raise ValueError("Connect the storyboard from 'Narrate Storyboard' (it has no narration yet)")
        rate = storyboard["sample_rate"]
        rng = random.Random(seed)
        _free_gpu()
        device = comfy.model_management.get_torch_device()
        images = storyboard["images"]
        title = storyboard.get("title", "")
        reel = _is_reel(storyboard)
        closing = storyboard.get("outro_audio")
        if reel:
            # 9:16, short intro, pauses, transitions and follow card.
            width, height = REEL_SIZE
            intro_seconds = min(intro_seconds, REEL_INTRO)
            outro_seconds = min(outro_seconds, REEL_OUTRO)
            pause_before, pause_after = REEL_PAUSES
            fade = min(fade, REEL_FADE)
        limit = _limit(storyboard)
        if limit > 0:
            # The pauses squeezed if the video would still run over its limit.
            n = len(storyboard["audio"])
            spoken_s = sum(len(c) for c in storyboard["audio"]) / rate
            closing_s = len(closing) / rate if closing is not None else 0.0
            tail = 1.1 if reel else 1.6
            length = (intro_seconds + spoken_s + n * (pause_before + pause_after)
                      + (max(outro_seconds, closing_s + tail) if outro_seconds > 0 else 0))
            if length > limit:
                squeeze = min(length - limit + 0.3, n * max(pause_before + pause_after - 0.3, 0))
                pause_before, pause_after = (pause_before - squeeze / n * 0.4, pause_after - squeeze / n * 0.6)
                if length - squeeze > limit:
                    print(f"[documentary] the video is {length - squeeze:.0f} s: fewer photos or words would "
                          f"bring it under {limit:.0f} s")

        # The timeline: scenes of whole frames, each with its audio padded to
        # exactly the same length, so picture and sound stay in sync.
        scenes, audio, speech, spoken = [], [], [], []  # spoken: (text, start, end) in seconds

        def add(scene, frames, clip=None, lead=0.0, timing=None, text=""):
            start = sum(len(a) for a in audio)
            total = round(frames / fps * rate)
            seg = np.zeros(total, dtype=np.float32)
            mask = np.zeros(total, dtype=np.uint8)
            if clip is not None:
                lead = round(lead * rate)
                seg[lead:lead + len(clip)] = clip[:max(total - lead, 0)]
                mask[lead:lead + len(clip)] = 1
                for part, s0, s1 in (timing or [(text, 0, len(clip))]):
                    spoken.append((part, (start + lead + s0) / rate, (start + lead + s1) / rate))
            scenes.append([scene, frames])
            audio.append(seg)
            speech.append(mask)

        if intro_seconds > 0:
            add(_IntroScene(images, title, MOOD_LABELS.get(storyboard.get("mood"), ""), width, height,
                            intro_seconds, rng, device), round(intro_seconds * fps))
        timings = storyboard.get("timings") or [None] * len(storyboard["audio"])
        details = storyboard.get("details") or [""] * len(images)
        for index, (image, clip, timing, text, detail) in enumerate(zip(images, storyboard["audio"], timings,
                                                                       storyboard["texts"], details)):
            frames = math.ceil((pause_before + len(clip) / rate + pause_after) * fps)
            move = motion
            if motion == STORY_MOTION:
                moves = storyboard.get("motions") or []
                move = moves[index] if index < len(moves) and moves[index] in MOVES else "alternate"
            add(_PhotoScene(image, index, move, zoom, rng, width, height, device, tall=reel, details=detail),
                frames, clip, pause_before, timing, text)
        if outro_seconds > 0:
            tail = 1.1 if reel else 1.6
            seconds = max(outro_seconds, len(closing) / rate + tail if closing is not None else 0)
            frames = math.ceil(seconds * fps)
            add(_OutroScene(images[-1], title, follow_text.strip(), handle.strip(), width, height, frames / fps,
                            device), frames, closing, 0.4 if reel else 0.5, storyboard.get("outro_timings"),
                storyboard.get("outro_text", ""))

        sound, sound_rate = np.concatenate(audio), rate
        if storyboard.get("music") is not None and music_volume > 0:
            sound, sound_rate = _mix(sound, rate, np.concatenate(speech), storyboard["music"],
                                     storyboard["music_rate"], music_volume, music_duck)

        # Transitions: centred on each cut, at most a third of either scene.
        starts = np.cumsum([0] + [f for _, f in scenes]).tolist()
        half = round(fade * fps / 2)
        kinds, last = [], None
        for b in range(1, len(scenes)):
            if transition != "mix":
                kind = transition
            elif b == 1 and intro_seconds > 0:
                kind = "zoom through"
            elif b == len(scenes) - 1 and outro_seconds > 0:
                kind = "dissolve"
            else:
                kind = rng.choice([k for k in MIX if k != last])
            last = kind
            kinds.append((kind, min(half, scenes[b - 1][1] // 3, scenes[b][1] // 3)))
        h_in = [0] + [h for _, h in kinds]
        h_out = [h for _, h in kinds] + [0]
        fx = _Transitions(width, height, device)

        def render(j, g):
            # Scene j at global frame g: u runs 0..1 over all the time it is visible.
            scene, frames = scenes[j]
            first, span = starts[j] - h_in[j], frames + h_in[j] + h_out[j]
            return scene.frame(min(max((g - first) / max(span - 1, 1), 0), 1), (g - starts[j]) / fps)

        # Subtitles: at the bottom; in a reel above Instagram's caption and left
        # of its buttons, just under the photo.
        cue_x, cue_y, cue_w = (width * 0.46, height * 0.8, width * 0.8) if reel else (width / 2, height * 0.9,
                                                                                       width * 0.84)
        unit = min(width, height)
        cues = []
        if subtitles:
            font = _font("sans", unit * (0.055 if reel else 0.048) * subtitle_size)
            for text, s0, s1 in spoken:
                cues += _subtitle_cues(text, s0, s1, font, cue_w)
            stroke = max(2, round(unit * 0.0045 * subtitle_size))
            cues = [(s0, s1, _text_layer(lines, font, device, stroke=stroke, shadow=stroke)) for s0, s1, lines in cues]

        out_dir = folder_paths.get_temp_directory()
        os.makedirs(out_dir, exist_ok=True)
        fd, video_path = tempfile.mkstemp(prefix="documentary_", suffix=".mp4", dir=out_dir)
        os.close(fd)
        with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
            wav = os.path.join(tmp, "narration.wav")
            _write_wav(wav, sound, sound_rate)
            cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                   "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
                   "-i", wav,
                   "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
                   "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", video_path]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                with torch.no_grad():
                    j = 0
                    for g in range(starts[-1]):
                        comfy.model_management.throw_exception_if_processing_interrupted()
                        while g >= starts[j + 1]:
                            j += 1
                            if j >= 2:
                                scenes[j - 2][0].release()
                        if j > 0 and g < starts[j] + h_in[j]:
                            kind, h = kinds[j - 1]
                            frame = fx(kind, render(j - 1, g), render(j, g), (g - starts[j] + h + 0.5) / (2 * h))
                        elif j < len(scenes) - 1 and g >= starts[j + 1] - h_out[j]:
                            kind, h = kinds[j]
                            frame = fx(kind, render(j, g), render(j + 1, g), (g - starts[j + 1] + h + 0.5) / (2 * h))
                        else:
                            frame = render(j, g)
                        frame = frame.clamp(0, 1).contiguous()
                        t = g / fps
                        for s0, s1, layer in cues:
                            if s0 <= t < s1:
                                # A short fade in and out, so captions don't pop.
                                _paste(frame, layer, cue_x, cue_y - layer.shape[1] / 2,
                                       min((t - s0) / 0.12, (s1 - t) / 0.12, 1))
                        rgb = (frame[0].permute(1, 2, 0) * 255).round().to(torch.uint8).cpu().numpy()
                        proc.stdin.write(rgb.tobytes())
            except BrokenPipeError:
                pass
            finally:
                for scene, _ in scenes:
                    scene.release()
                proc.stdin.close()
                err = proc.stderr.read().decode(errors="replace")
                if proc.wait() != 0:
                    raise RuntimeError(f"ffmpeg failed: {err[-1000:]}")
        return (InputImpl.VideoFromFile(video_path),)


def _best_photo(images):
    """Index of the sharpest, most colourful image."""
    best, pick = -1.0, 0
    lap = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float32)
    for i, image in enumerate(images):
        x = image[0].permute(2, 0, 1)[None].float().cpu()
        x = F.interpolate(x, size=(384, max(round(384 * x.shape[3] / x.shape[2]), 1)), mode="area")
        sharp = F.conv2d(x.mean(1, keepdim=True), lap).var().item()
        r, g, b = x[0]
        rg, yb = r - g, (r + g) / 2 - b
        colour = math.hypot(rg.std().item(), yb.std().item()) + 0.3 * math.hypot(rg.mean().item(), yb.mean().item())
        score = math.sqrt(sharp) * (0.2 + colour)
        if score > best:
            best, pick = score, i
    return pick


class DocumentaryThumbnail:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "storyboard": (STORYBOARD,),
                "width": ("INT", {"default": 1920, "min": 256, "max": 3840, "step": 2,
                                  "tooltip": "For a YouTube video; a reel's cover is always 1080x1920"}),
                "height": ("INT", {"default": 1080, "min": 256, "max": 3840, "step": 2,
                                   "tooltip": "For a YouTube video; a reel's cover is always 1080x1920"}),
                "photo": ("INT", {"default": 0, "min": 0, "max": MAX_IMAGES,
                                  "tooltip": "Which photo (1, 2, ...); 0 = the story's cover choice (a reel "
                                             "format such as 'mini-documentary'), else the sharpest, most colourful one"}),
                "headline": ("STRING", {"default": "",
                                        "placeholder": "Optional: the text on the thumbnail. Empty: the story's hook"}),
                "label": ("STRING", {"default": "", "placeholder": "Optional: the small label. Empty: from the mood"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("thumbnail",)
    FUNCTION = "run"
    CATEGORY = "documentary"

    def run(self, storyboard, width, height, photo, headline, label):
        # It only needs the story, so it can run right after it, while the
        # writer still fills the GPU.
        _free_gpu()
        images = storyboard["images"]
        cover = storyboard.get("cover")
        index = photo - 1 if 1 <= photo <= len(images) else \
            cover - 1 if isinstance(cover, int) and 1 <= cover <= len(images) else _best_photo(images)
        device = comfy.model_management.get_torch_device()
        reel = _is_reel(storyboard)
        if reel:
            width, height = REEL_SIZE
        with torch.no_grad():
            if reel:
                # The square photo high up; the text below it, inside the 3:4
                # middle that the profile grid shows.
                frame = _framed(images[index], width, height, 1.0, device, centre=0.38)
            else:
                frame = _cover(images[index], width, height, 1.0, device)
            grey = frame.mean(1, keepdim=True)
            frame = ((grey + (frame - grey) * 1.15 - 0.5) * 1.08 + 0.5).clamp(0, 1)
            # Darker towards the bottom left, where the text goes.
            ys = torch.linspace(0, 1, height, device=device)[:, None]
            xs = torch.linspace(0, 1, width, device=device)[None, :]
            shade = torch.maximum(((ys - 0.35) / 0.65).clamp(0, 1) ** 1.4 * 0.85,
                                  ((0.65 - xs) / 0.65).clamp(0, 1) ** 1.6 * (0.0 if reel else 0.5))
            frame = (frame * (1 - shade)).contiguous()
            text = headline.strip() or storyboard.get("hook") or storyboard.get("title") or ""
            margin = round(min(width, height) * 0.06)
            bottom = height * 0.85 if reel else height - margin
            if text:
                font, lines = (_fit(text, "sans", width * 0.13, width * 0.88, 3) if reel else
                               _fit(text, "sans", min(height * 0.14, width * 0.09), width * 0.62, 3))
                stroke, shadow = max(2, round(font.size * 0.03)), round(font.size * 0.06)
                layer = _text_layer(lines, font, device, stroke=stroke, shadow=shadow, align="left", spacing=0.02)
                pad = stroke + 3 * shadow + 2  # _text_layer's margin around the text
                _paste(frame, layer, margin - pad + layer.shape[2] / 2, bottom + pad - layer.shape[1] / 2)
                bottom -= layer.shape[1] - 2 * pad + height * 0.03
            unit = min(width, height)
            bar = torch.zeros(4, max(round(unit * 0.012), 3), round(unit * 0.125), device=device)
            bar[:3] = torch.tensor([c / 255 for c in ACCENT], device=device)[:, None, None]
            bar[3] = 1
            _paste(frame, bar, margin + bar.shape[2] / 2, bottom - bar.shape[1] / 2)
            bottom -= bar.shape[1] + unit * 0.02
            label = (label.strip() or MOOD_LABELS.get(storyboard.get("mood"), "")).upper()
            if label:
                pill = _pill(label, _font("sans", unit * 0.03), device, colors=((0, 0, 0), (0, 0, 0)), alpha=150)
                _paste(frame, pill, margin + pill.shape[2] / 2, bottom - pill.shape[1] / 2)
        return (frame.permute(0, 2, 3, 1).cpu(),)


class DocumentarySocialPost:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "storyboard": (STORYBOARD,),
                "writer_model": ("STRING", {"default": "qwen3:32b"}),
                "handle": ("STRING", {"default": "", "placeholder": "Optional: your @handle"}),
                "extra": ("STRING", {"multiline": True, "default": "",
                                     "placeholder": "Optional: anything to mention (place, gear, a call to action)"}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.5, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("post",)
    FUNCTION = "run"
    CATEGORY = "documentary"

    def run(self, storyboard, writer_model, handle, extra, temperature, seed):
        import re

        handle = handle.strip()
        if handle and not handle.startswith("@"):
            handle = f"@{handle}"
        style = storyboard.get("custom_style") or storyboard.get("mood", "")
        language = storyboard.get("language", "English")
        about = {
            "title": storyboard.get("title", ""), "tone": style, "subjects": storyboard.get("subjects", []),
            "background": storyboard.get("brief", ""), "narration": storyboard.get("texts", []),
        }
        youtube = not _is_reel(storyboard)
        place = "YouTube channel" if youtube else "Instagram account"
        ask = "subscribe" if youtube else "follow"
        prompt = (
            f"You run the {place} that posts this short narrated video, in {language}. "
            f"The video: {json.dumps(about, ensure_ascii=False)}. "
            + (f"Also mention: {extra.strip()} " if extra.strip() else "")
            + "Write the post as separate parts: "
            + ("'title': the YouTube title, at most 70 characters, curious and specific, with the main key phrase "
               "early. " if youtube else "")
            + "'hook': a scroll-stopping first line of at most 10 words. "
            "'body': two or three short paragraphs, 50 to 90 words in all, separated by blank lines, in the video's "
            "tone, that tease the story without retelling it, with true details only from the narration, two or "
            "three fitting emojis, and two or three of the key phrases worked in naturally. "
            "'question': one short question to the viewer, to get comments. "
            f"'cta': one short line inviting the viewer to {ask} for more stories like this. No hashtags in any of "
            "these. "
            + (f"'hashtags': exactly {3 if youtube else 5}, a mix of specific (the subjects, the place) and popular "
               "ones for this kind of video, no spaces, without the # sign. ")
            + ("'tags': 12 YouTube tags, from specific to broad, each 1 to 4 words. " if youtube else "")
            + "'keyphrases': 8 search key phrases of 2 to 5 words that people would type to find this video. "
            "'alt_text': one sentence describing the video for people who can't see it. "
            "Answer only with JSON with exactly these keys."
        )
        fields = ("title", "hook", "body", "question", "cta", "alt_text") if youtube else (
            "hook", "body", "question", "cta", "alt_text")
        count = 3 if youtube else 5
        lists = {"hashtags": {"type": "array", "items": {"type": "string"}, "minItems": count, "maxItems": count},
                 "keyphrases": {"type": "array", "items": {"type": "string"}}}
        if youtube:
            lists["tags"] = {"type": "array", "items": {"type": "string"}}
        schema = {
            "type": "object",
            "properties": {**{k: {"type": "string"} for k in fields}, **lists},
            "required": [*fields, *lists],
        }
        for attempt in range(3):
            post = _llm_json(writer_model.strip(), prompt, schema, temperature, seed + attempt)
            if len(str(post.get("body", "")).split()) >= 25:
                break
        tags = []
        for tag in post.get("hashtags", []):
            tag = re.sub(r"[^\w]", "", str(tag), flags=re.UNICODE)
            if tag and tag.lower() not in [t.lower() for t in tags]:
                tags.append(tag)
        hashtags = " ".join(f"#{t}" for t in tags[:count])
        cta = str(post.get("cta", "")).strip() or f"{ask.capitalize()} for more stories like this."
        if handle and handle.lower() not in cta.lower():
            cta = f"{cta.rstrip('.!')} \u2192 {handle}"
        caption = "\n\n".join(x for x in (str(post.get(k, "")).strip() for k in ("hook", "body", "question")) if x)
        caption = f"{caption}\n\n{cta}"
        phrases = [str(p).strip() for p in post.get("keyphrases", []) if str(p).strip()]
        alt = str(post.get("alt_text", "")).strip()
        if youtube:
            title = str(post.get("title", "")).strip()[:100]
            yt_tags = ", ".join(str(t).strip().lstrip("#") for t in post.get("tags", []) if str(t).strip())
            text = (f"=== Title ===\n{title}\n\n=== Description (ready to paste) ===\n{caption}\n\n{hashtags}\n\n"
                    f"=== Tags (paste into YouTube Studio) ===\n{yt_tags}\n\n"
                    f"=== SEO key phrases ===\n" + "\n".join(f"- {p}" for p in phrases))
        else:
            text = (f"=== Ready to paste ===\n{caption}\n\n{hashtags}\n\n"
                    f"=== SEO key phrases ===\n" + "\n".join(f"- {p}" for p in phrases)
                    + f"\n\n=== Alt text ===\n{alt}")
        return (text,)


# Stored files: listed and deleted from the "Documentary → Stored files" menu
# entry (web/storage.js). Only inside ComfyUI's output, input and temp folders.
STORAGE_FOLDERS = {
    "output": folder_paths.get_output_directory,
    "input": folder_paths.get_input_directory,
    "temp": folder_paths.get_temp_directory,
}


def _storage_list():
    files = []
    for folder, base_of in STORAGE_FOLDERS.items():
        base = base_of()
        for root, dirs, names in os.walk(base):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for name in names:
                if name.startswith("."):
                    continue
                full = os.path.join(root, name)
                try:
                    st = os.stat(full)
                except OSError:
                    continue
                rel = os.path.relpath(full, base).replace(os.sep, "/")
                files.append({"folder": folder, "path": rel, "size": st.st_size, "mtime": st.st_mtime})
    files.sort(key=lambda f: f["mtime"], reverse=True)
    usage = shutil.disk_usage(folder_paths.get_output_directory())
    return {"files": files, "disk": {"total": usage.total, "free": usage.free}}


def _storage_delete(items):
    deleted, freed, errors = 0, 0, []
    for item in items:
        folder, rel = item.get("folder"), str(item.get("path", ""))
        if folder not in STORAGE_FOLDERS or not rel:
            errors.append(f"{folder}/{rel}: not allowed")
            continue
        base = os.path.realpath(STORAGE_FOLDERS[folder]())
        full = os.path.realpath(os.path.join(base, rel))
        if not full.startswith(base + os.sep) or not os.path.isfile(full):
            errors.append(f"{folder}/{rel}: not a file in {folder}")
            continue
        try:
            size = os.path.getsize(full)
            os.remove(full)
            deleted, freed = deleted + 1, freed + size
        except OSError as e:
            errors.append(f"{folder}/{rel}: {e.strerror}")
            continue
        # Remove subfolders left empty (e.g. output/video), never the folder itself.
        parent = os.path.dirname(full)
        while parent != base and parent.startswith(base + os.sep):
            try:
                os.rmdir(parent)
            except OSError:
                break
            parent = os.path.dirname(parent)
    return {"deleted": deleted, "freed": freed, "errors": errors}


try:
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.get("/documentary/storage")
    async def _storage_get(request):
        return web.json_response(_storage_list())

    @PromptServer.instance.routes.post("/documentary/storage/delete")
    async def _storage_post(request):
        body = await request.json()
        return web.json_response(_storage_delete(body.get("files", [])))
except (ImportError, AttributeError):
    pass  # not running inside ComfyUI's server

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "DocumentaryPhoto": DocumentaryPhoto,
    "DocumentaryStory": DocumentaryStory,
    "DocumentaryNarrate": NarrateStoryboard,
    "DocumentaryMusic": DocumentaryMusic,
    "DocumentaryKenBurnsVideo": KenBurnsVideo,
    "DocumentarySocialPost": DocumentarySocialPost,
    "DocumentaryThumbnail": DocumentaryThumbnail,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DocumentaryPhoto": "Documentary Photo",
    "DocumentaryStory": "Documentary Story",
    "DocumentaryNarrate": "Narrate Storyboard",
    "DocumentaryMusic": "Documentary Music (ACE-Step 1.5)",
    "DocumentaryKenBurnsVideo": "Ken Burns Video",
    "DocumentarySocialPost": "Social Post",
    "DocumentaryThumbnail": "Documentary Thumbnail",
}
