"""Species identification for Documentary Story: BioCLIP 2 (imageomics, MIT;
trained on the Tree of Life 200M images) ranks the photo against the birds and
mammals recorded in the photo's country (from GBIF), plus a few labels for
"not an animal". A general vision model can't tell close species apart; this
can, and says how sure it is. Weights, species lists (with each species'
Wikipedia intro, the story's facts) and their text embeddings are kept in the
models volume (models/bioclip).

Nothing of the photos or their notes leaves the homelab: the classifier runs
here, the weights load from the volume (Story's _local_first), and the only
requests out are one-time downloads: the weights, and a country's species list
and facts (GBIF and Wikipedia, asked by country code and scientific names).
Once those are on the volume, it works without the internet."""
import concurrent.futures
import json
import os
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import torch
from PIL import Image

import folder_paths

MODEL = "hf-hub:imageomics/bioclip-2"
GBIF = "https://api.gbif.org/v1"
# GBIF class keys: the groups a species list holds.
GROUPS = {"Aves": 212, "Mammalia": 359}
# A species needs this many records in the country (leaves out one-off vagrants).
MIN_RECORDS = 25
# How much a species' commonness in the country counts: the classifier's
# probability times (share of the records) ** PRIOR. Close look-alikes (a
# northern wheatear and a rare pied one) then go to the common one, unless the
# photo clearly shows the rare one.
PRIOR = 0.25
DEFAULT_COUNTRY = "SE"
# When one of these wins, the photo has no animal to name.
NOT_ANIMALS = ["a landscape", "a person", "people", "a building", "a city street", "a plant", "a flower", "a tree",
               "an insect", "a fish", "a reptile", "the sky", "water", "food", "a vehicle", "a boat"]
# Wikimedia throttles clients without a contact URL in the User-Agent.
HEADERS = {"User-Agent": "comfyui-documentary/1.0 (https://github.com/kaustavgithub/comfyui-image)"}

_state = {}


def _dir():
    path = os.path.join(folder_paths.models_dir, "bioclip")
    os.makedirs(path, exist_ok=True)
    return path


def _get(path, timeout=60):
    with urllib.request.urlopen(urllib.request.Request(f"{GBIF}{path}", headers=HEADERS), timeout=timeout) as r:
        return json.load(r)


def country_code(name):
    """An ISO code for a country name (as Immich writes it), or the name if it
    already is one; "" if unknown."""
    name = (name or "").strip()
    if len(name) == 2 and name.isalpha():
        return name.upper()
    if not name:
        return ""
    if "countries" not in _state:
        path = os.path.join(_dir(), "countries.json")
        if not os.path.exists(path):
            with open(path, "w") as f:
                json.dump({c["title"].lower(): c["iso2"] for c in _get("/enumeration/country")}, f)
        with open(path) as f:
            _state["countries"] = json.load(f)
    return _state["countries"].get(name.lower(), "")


def _wiki(query, tries=5):
    """A Wikipedia API answer; waits and asks again when told to slow down."""
    import time

    for attempt in range(tries):
        req = urllib.request.Request(f"https://en.wikipedia.org/w/api.php?{query}", headers=HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)["query"]
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == tries - 1:
                raise
            time.sleep(float(e.headers.get("Retry-After") or 0) or 5 * 2 ** attempt)


def _wiki_intros(titles, limit=1200):
    """The intro of each Wikipedia article, as plain text (20 per request)."""
    out = {}
    for i in range(0, len(titles), 20):
        batch = titles[i:i + 20]
        q = urllib.parse.urlencode({"action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1,
                                    "exlimit": 20, "titles": "|".join(batch), "format": "json"})
        for page in _wiki(q).get("pages", {}).values():
            text = (page.get("extract") or "").strip()
            if text:
                out[page["title"]] = text[:limit]
    return out


def _wiki_titles(names):
    """The English Wikipedia article each scientific name redirects to: its
    common name as birders use it (IOC), e.g. Bucephala clangula -> Common
    goldeneye. 50 names per request (Wikipedia refuses many small ones)."""
    out = {}
    for i in range(0, len(names), 50):
        batch = names[i:i + 50]
        q = urllib.parse.urlencode({"action": "query", "titles": "|".join(batch), "redirects": 1,
                                    "format": "json"})
        reply = _wiki(q)
        step = {x["from"]: x["to"] for x in reply.get("normalized", []) + reply.get("redirects", [])}
        found = {p["title"] for p in reply.get("pages", {}).values() if "missing" not in p}
        for name in batch:
            title = name
            while title in step:
                title = step[title]
            if title in found:
                out[name] = title
    return out


def _species(item):
    key, records = item
    s = _get(f"/species/{key}")
    scientific = s.get("species") or s.get("canonicalName", "")
    if not scientific or " x " in scientific or "×" in scientific:  # hybrids
        return None
    names = _get(f"/species/{key}/vernacularNames?limit=200").get("results", [])
    # GBIF's English names without ringing codes (GRTI, COPO); species_list()
    # puts the Wikipedia title first.
    common = []
    for n in names:
        v = (n.get("vernacularName") or "").strip()
        if n.get("language") == "eng" and v and not (v.isupper() and len(v) <= 5) \
                and v.lower() not in (c.lower() for c in common):
            common.append(v)
    taxon = " ".join(s.get(r, "") for r in ("kingdom", "phylum", "class", "order", "family")).split()
    return {"key": key, "scientific": scientific, "taxon": " ".join(taxon), "common": common, "records": records}


def species_list(country):
    """The birds and mammals recorded in a country, cached in models/bioclip:
    [{"scientific", "taxon", "common": [English names, the Wikipedia title
    first], "records"}]."""
    key = f"species_{country}"
    if key in _state:
        return _state[key]
    path = os.path.join(_dir(), f"species_{country}.json")
    if os.path.exists(path):
        with open(path) as f:
            found = json.load(f)
        if all("facts" in s for s in found):  # older lists have no facts: download again
            _state[key] = found
            return found
    keys = []
    for key in GROUPS.values():
        q = urllib.parse.urlencode({"country": country, "classKey": key, "limit": 0, "facet": "speciesKey",
                                    "facetLimit": 3000, "facetMincount": MIN_RECORDS})
        facets = _get(f"/occurrence/search?{q}").get("facets", [])
        keys += [(int(c["name"]), int(c["count"])) for f in facets for c in f.get("counts", [])]
    print(f"[documentary] species list {country}: {len(keys)} species from GBIF and Wikipedia (once)")
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        found = [s for s in pool.map(_species, keys) if s]
    titles = _wiki_titles([s["scientific"] for s in found])
    intros = _wiki_intros(sorted(set(titles.values())))
    for s in found:
        title = titles.get(s["scientific"], "")
        if title and title.lower() != s["scientific"].lower():
            s["common"] = [title] + [c for c in s["common"] if c.lower() != title.lower()]
        s["facts"] = f"{title}: {intros[title]}" if title in intros else ""
    found = [s for s in found if s["common"]]
    with open(path + ".tmp", "w") as f:
        json.dump(found, f, ensure_ascii=False)
    os.replace(path + ".tmp", path)
    _state[key] = found
    return found


def _model(device):
    if "model" not in _state:
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(MODEL, cache_dir=_dir())
        # The tokenizer's config too from models/bioclip, not Hugging Face's default cache.
        _state.update(model=model.eval(), preprocess=preprocess,
                      tokenizer=open_clip.get_tokenizer(MODEL, cache_dir=_dir()))
    _state["model"].to(device)
    return _state["model"]


def _labels(country, device):
    """Text embeddings of the country's species, then of NOT_ANIMALS."""
    key = f"labels_{country}"
    if key not in _state:
        species = species_list(country)
        path = os.path.join(_dir(), f"labels_{country}.pt")
        if os.path.exists(path):
            text = torch.load(path)
        else:
            model = _model(device)
            prompts = [f"a photo of {s['taxon']} {s['scientific']} with common name {s['common'][0]}." for s in species]
            prompts += [f"a photo of {x}." for x in NOT_ANIMALS]
            parts = []
            with torch.no_grad():
                for i in range(0, len(prompts), 256):
                    tokens = _state["tokenizer"](prompts[i:i + 256]).to(device)
                    parts.append(torch.nn.functional.normalize(model.encode_text(tokens).float(), dim=-1).cpu())
            text = torch.cat(parts)
            torch.save(text, path)
        records = torch.tensor([float(s["records"]) for s in species])
        prior = torch.cat([(records / records.sum()) ** PRIOR, torch.ones(len(NOT_ANIMALS))])
        _state[key] = (species, text, prior)
    return _state[key]


def _views(image):
    """The whole photo padded to a square, and its centre: a subject at the
    edge is not cut off, and a small one is seen larger."""
    arr = (image.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    pil = Image.fromarray(arr)
    side = max(pil.size)
    square = Image.new("RGB", (side, side), (0, 0, 0))
    square.paste(pil, ((side - pil.width) // 2, (side - pil.height) // 2))
    return [square, pil]


def identify(images, countries, device):
    """For each image ([H,W,3] tensor) and its country code: (common name,
    scientific name, probability) of the best match, or None when the photo
    shows no animal; plus the runner-up (for the log)."""
    out = []
    try:
        model = _model(device)
        for image, country in zip(images, countries):
            species, text, prior = _labels(country or DEFAULT_COUNTRY, device)
            views = torch.stack([_state["preprocess"](v) for v in _views(image)]).to(device)
            with torch.no_grad():
                feats = torch.nn.functional.normalize(model.encode_image(views).float(), dim=-1).mean(0)
                feats = torch.nn.functional.normalize(feats, dim=-1).cpu()
                probs = (model.logit_scale.exp().cpu() * feats @ text.T).softmax(-1)
            # "Not an animal" labels keep their own probability; the species share the rest by prior.
            n = len(species)
            animal = probs[:n] * prior[:n]
            probs = torch.cat([animal / animal.sum() * probs[:n].sum(), probs[n:]])
            top = probs.topk(2)
            picks = []
            for p, i in zip(top.values.tolist(), top.indices.tolist()):
                if i >= len(species):
                    picks.append((None, NOT_ANIMALS[i - len(species)], p))
                else:
                    picks.append((species[i]["common"][0], species[i]["scientific"], p))
            out.append(picks)
    finally:
        if "model" in _state:
            _state["model"].to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def facts(name, country):
    """The Wikipedia intro of a species, by its scientific or an English name
    (exact, ignoring case), from the local list; "" if it isn't one."""
    name = (name or "").strip().lower()
    if not name:
        return ""
    for s in species_list(country or DEFAULT_COUNTRY):
        if name == s["scientific"].lower() or name in (c.lower() for c in s["common"]):
            return s.get("facts", "")
    return ""


def named_in(text, country):
    """The species a text names (an English or scientific name, as a whole
    phrase, ignoring case; the longest if several): (common, scientific) or
    None. For the photographer's own note, which then beats the classifier."""
    import re

    if not text:
        return None
    best, size = None, 0
    for s in species_list(country or DEFAULT_COUNTRY):
        for name in s["common"] + [s["scientific"]]:
            if len(name) > size and re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", text, re.IGNORECASE):
                best, size = (s["common"][0], s["scientific"]), len(name)
    return best
