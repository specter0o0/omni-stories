#!/usr/bin/env python3
"""
Omni Stories: Automated Reddit Story Video Generation Engine

Production-ready video generation system that transforms Reddit stories into
narrated videos with synchronized captions, background footage, and thumbnails.
Supports both cloud-based (ElevenLabs) and local (Kokoro) TTS engines.
"""

import os
import sys
import json
import random
import argparse
import subprocess
import re
import shutil
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv, set_key

try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
except ImportError:
    Image = None

try:
    import yaml
except ImportError:
    yaml = None

try:
    import requests
except ImportError:
    requests = None

try:
    import yt_dlp
except ImportError:
    yt_dlp = None

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

VERSION = "1.1.0"

class TerminalUI:
    """ANSI-based terminal output formatting and interaction"""
    
    CYAN = "\033[0;36m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    RED = "\033[0;31m"
    PINK = "\033[95m"
    BOLD = "\033[1m"
    NC = "\033[0m"
    MAGENTA = "\033[0;35m"
    BLUE = "\033[0;34m"
    WHITE = "\033[1;37m"
    
    SYM_OK = f"{GREEN}✔{NC}"
    SYM_WARN = f"{YELLOW}⚠{NC}"
    SYM_ERR = f"{RED}✖{NC}"
    SYM_INFO = f"{CYAN}ℹ{NC}"
    
    ASCII_ART = f"""
{CYAN}╔═══════════════════════════════════════════════════════════╗
║  {MAGENTA} ██████╗ ███╗   ███╗███╗   ██╗██╗{CYAN}                        ║
║  {MAGENTA}██╔═══██╗████╗ ████║████╗  ██║██║{CYAN}                        ║
║  {MAGENTA}██║   ██║██╔████╔██║██╔██╗ ██║██║{CYAN}                        ║
║  {MAGENTA}██║   ██║██║╚██╔╝██║██║╚██╗██║██║{CYAN}                        ║
║  {MAGENTA}╚██████╔╝██║ ╚═╝ ██║██║ ╚████║██║{CYAN}                        ║
║  {MAGENTA} ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═══╝╚═╝{CYAN}                        ║
║  {WHITE}███████╗████████╗ ██████╗ ██████╗ ██╗███████╗███████╗{CYAN}    ║
║  {WHITE}██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗██║██╔════╝██╔════╝{CYAN}    ║
║  {WHITE}███████╗   ██║   ██║   ██║██████╔╝██║█████╗  ███████╗{CYAN}    ║
║  {WHITE}╚════██║   ██║   ██║   ██║██╔══██╗██║██╔══╝  ╚════██║{CYAN}    ║
║  {WHITE}███████║   ██║   ╚██████╔╝██║  ██║██║███████╗███████║{CYAN}    ║
║  {WHITE}╚══════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚═╝╚══════╝╚══════╝{CYAN}    ║
╚═══════════════════════════════════════════════════════════╝{NC}
"""

    @staticmethod
    def print_banner():
        """Display compact version header"""
        print(f"\n{TerminalUI.CYAN}{TerminalUI.BOLD}Omni-Stories v{VERSION}{TerminalUI.NC}")
        print(f"{TerminalUI.CYAN}{'-' * (16 + len(VERSION))}{TerminalUI.NC}\n")

    @staticmethod
    def print_large_banner():
        """Display ASCII art banner"""
        print(TerminalUI.ASCII_ART)

    @staticmethod
    def notify(text: str, color: Optional[str] = None):
        """Print color-coded message"""
        c = color if color else ""
        print(f"{c}{text}{TerminalUI.NC}")

    @staticmethod
    def input_prompt(text: str) -> str:
        """Display input prompt with visual indicator"""
        return input(f"{TerminalUI.YELLOW}{text} {TerminalUI.NC}")


# Directory Hierarchy:
# root/
#   .omni-stories-data/ (ENGINE_DIR: main.py, models, internal assets)
#   config.yaml         (User Config)
#   .env               (User Credentials)
#   output/            (User Productions)

ENGINE_DIR = Path(__file__).resolve().parent
BASE_DIR = ENGINE_DIR.parent

CONFIG_PATH = BASE_DIR / "config.yaml"
ENV_PATH = BASE_DIR / ".env"

DEFAULT_SYSTEM_PATHS = {
    "data": str(ENGINE_DIR.relative_to(BASE_DIR)),
    "output": "output",
    "background_videos": f"{ENGINE_DIR.relative_to(BASE_DIR)}/background_videos",
    "models": f"{ENGINE_DIR.relative_to(BASE_DIR)}/models",
    "tts_generations": f"{ENGINE_DIR.relative_to(BASE_DIR)}/tts_generations",
    "database": f"{ENGINE_DIR.relative_to(BASE_DIR)}/database.json",
    "kokoro": f"{ENGINE_DIR.relative_to(BASE_DIR)}/models/kokoro"
}

APP_DEFAULTS = {
    "quality": "1080p",
    "aspect_ratio": "16:9",
    "background_fit": "crop",
    "background_volume": "0%",
    "narration_volume": "100%",

    "thumbnail_bg_blur": "50%",
    "thumbnail_font": "Arial Black",
    "thumbnail_font_size": 125,
    "thumbnail_font_color": "#FFFFFF",
    "thumbnail_font_shadow": True,
    "thumbnail_shadow_color": "#FFFFFF",
    "thumbnail_strength": "50%",
    "thumbnail_glow": True,
    "thumbnail_glow_color": "#FFFFFF",
    "thumbnail_glow_strength": "75%",

    "captions": True,
    "caption_max_words_per_line": 4,
    "caption_highlight_color": "#f8d568",
    "caption_font": "Arial Black",
    "caption_font_size": 100,
    "caption_position": "center",
    "caption_font_color": "#FFFFFF",
    "caption_font_shadow": True,
    "caption_shadow_color": "#FFFFFF",
    "caption_strength": "40%",
    "caption_glow": True,
    "caption_glow_color": "#FFFFFF",
    "caption_glow_strength": "75%",
    "caption_glow_darken_factor": 0.7,

    "voice_model": "v2",
    "voice_id": "pNInz6obpgDQGcFmaJgB",
    "kokoro_voice": "am_adam",
    "max_db_entries": 500,
    "system_paths": DEFAULT_SYSTEM_PATHS
}

CAPTION_COST_WEIGHTS = {
    "overflow_penalty": 200,
    "period_bonus": 150,
    "comma_bonus": 50,
    "length_deviation": 30,
    "capitalization_penalty": 40,
    "gap_penalty": 1000
}

THUMBNAIL_CONFIG = {
    "max_chars_per_line": 20,
    "rotation_range": (-3, 3),
    "shadow_offset_divisor": 20,
    "line_spacing_multiplier": 1.4
}

VIDEO_CAPTURE_RANGE = (0.1, 0.9)
GAUSSIAN_BLUR_MULTIPLIER = 20
USER_AGENT = f"OmniStories/{VERSION}"


def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> None:
    """
    Recursively merge update dictionary into base dictionary.
    
    Args:
        base: The target dictionary to merge into.
        update: The source dictionary containing override values.
    """
    for k, v in update.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            deep_merge(base[k], v)
        else:
            base[k] = v

def get_config() -> Dict[str, Any]:
    """
    Load and resolve application configuration.
    
    Returns:
        A dictionary containing the merged result of defaults and custom config.
    """
    if yaml is None:
        return APP_DEFAULTS.copy()
    
    config = APP_DEFAULTS.copy()
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, 'r') as f:
                loaded = yaml.safe_load(f)
                if loaded and isinstance(loaded, dict):
                    deep_merge(config, loaded)
                    if "background_audio" in loaded and "background_volume" not in loaded:
                        config["background_volume"] = loaded["background_audio"]
        except Exception:
            pass
    return config


def get_resolved_paths() -> Dict[str, Path]:
    """
    Resolve absolute system paths based on configuration.
    
    Returns:
        Dictionary mapping path identifiers to absolute Path objects.
    """
    config = get_config()
    paths_raw = config.get("system_paths", DEFAULT_SYSTEM_PATHS)
    return {k: BASE_DIR / v for k, v in paths_raw.items()}


Paths = get_resolved_paths()
load_dotenv(ENV_PATH)


class KokoroEngine:
    """Provides local offline text-to-speech synthesis using the Kokoro model."""
    
    def __init__(self) -> None:
        """Initialize the ONNX inference session and load vocabulary."""
        TerminalUI.notify("Initializing Kokoro-TTS Engine...", TerminalUI.CYAN)
        try:
            import onnxruntime as ort
            import phonemizer
            
            model_path = Paths['kokoro'] / "model.onnx"
            vocab_path = Paths['kokoro'] / "tokenizer.json"
            
            self.session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
            with open(vocab_path, 'r') as f:
                self.vocab = json.load(f)['model']['vocab']
            self.voices = {}
            self.phonemizer = phonemizer
            TerminalUI.notify(f"{TerminalUI.SYM_OK} Kokoro-TTS Initialization Successful.")
            self.is_available = True
        except ImportError as e:
            TerminalUI.notify(f"Kokoro-TTS Skipped: Missing dependency ({e})", TerminalUI.YELLOW)
            self.is_available = False
        except Exception as e:
            TerminalUI.notify(f"Kokoro-TTS Initialization Failed: {e}", TerminalUI.RED)
            self.is_available = False

    def get_voice_embedding(self, name: str) -> Optional[Any]:
        """
        Load voice characteristics from binary embedding files.
        
        Args:
            name: Identifier for the requested voice.
            
        Returns:
            NumPy array containing the voice embedding data.
        """
        import numpy as np
        if name not in self.voices:
            path = Paths['kokoro'] / f"{name}.bin"
            if not path.exists():
                path = Paths['kokoro'] / "am_adam.bin"
            raw_data = np.fromfile(path, dtype=np.float32)
            self.voices[name] = raw_data.reshape(-1, 1, 256)
        return self.voices[name]

    def phonemize_text(self, text: str) -> str:
        """Convert plaintext to IPA phoneme strings."""
        from phonemizer import phonemize
        return phonemize(text, language='en-us', backend='espeak', strip=True)

    def tokenize_phonemes(self, phonemes: str) -> Any:
        """Map phoneme characters to model-specific token identifiers."""
        import numpy as np
        tokens = [0]
        for p in phonemes:
            if p in self.vocab:
                tokens.append(self.vocab[p])
            elif p == " ":
                tokens.append(16)
        tokens.append(0)
        return np.array([tokens], dtype=np.int64)

    def synthesize(self, text: str, voice_name: str, output_path: str) -> str:
        """
        Synthesize audio from text across multiple grammatical segments.
        
        Args:
            text: Input content to narrate.
            voice_name: Voice model identifier.
            output_path: Target filesystem location for the generated WAV file.
            
        Returns:
            The path to the finalized audio file.
        """
        import numpy as np
        import soundfile as sf
        
        sentences = re.split(r'([.!?])', text)
        chunks, current = [], ""
        for part in sentences:
            if len(current) + len(part) < 400:
                current += part
            else:
                if current.strip():
                    chunks.append(current)
                current = part
        if current.strip():
            chunks.append(current)

        audio_output = []
        for chunk in chunks:
            phonemes = self.phonemize_text(chunk)
            token_ids = self.tokenize_phonemes(phonemes)
            
            if token_ids.shape[1] > 510:
                token_ids = token_ids[:, :510]
                token_ids[0, -1] = 0
            
            voice_data = self.get_voice_embedding(voice_name)
            style_vector = voice_data[min(token_ids.shape[1]-2, voice_data.shape[0]-1)]
            
            audio = self.session.run(None, {
                "input_ids": token_ids,
                "style": style_vector,
                "speed": np.array([1.0], dtype=np.float32)
            })[0]
            audio_output.append(audio.squeeze())
        
        sf.write(output_path, np.concatenate(audio_output), 24000)
        return output_path


_global_kokoro = None


def generate_local_audio(text: str, output_path: str) -> str:
    """
    Generate audio using local Kokoro TTS engine.
    Initializes engine on first call and reuses for subsequent requests.
    """
    global _global_kokoro
    if _global_kokoro is None:
        _global_kokoro = KokoroEngine()
    
    if not _global_kokoro.is_available:
        TerminalUI.notify("Kokoro TTS unavailable. Cannot generate audio.", TerminalUI.RED)
        sys.exit(1)

    config = get_config()
    try:
        return _global_kokoro.synthesize(text, config.get("kokoro_voice", "am_adam"), output_path)
    except Exception as e:
        TerminalUI.notify(f"Kokoro Generation Failed: {e}", TerminalUI.RED)
        sys.exit(1)





def load_database() -> Dict[str, Any]:
    """
    Retrieve stored story metadata from the local database.
    
    Returns:
        The database contents as a dictionary.
    """
    db_path = Paths['database']
    if not db_path.exists():
        return {"stories": []}
    try:
        with open(db_path, 'r') as f:
            return json.load(f)
    except Exception:
        return {"stories": []}

def save_database(data: Dict[str, Any]) -> None:
    """
    Persist story metadata and enforce entry limits.
    
    Args:
        data: The database dictionary to save.
    """
    config = get_config()
    limit = config.get("max_db_entries", 500)
    data["stories"] = data.get("stories", [])[-limit:]
    with open(Paths['database'], 'w') as f:
        json.dump(data, f, indent=4)


def validate_cli_input(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    """
    Verify command-line arguments and prevent duplicate generation.
    
    Args:
        args: Parsed command-line arguments.
        config: Current application configuration.
    """
    reddit_regex = r'^https?://(?:www\.)?reddit\.com/r/[\w-]+/comments/[\w-]+/[\w-]+/?.*$'
    
    if getattr(args, 'new', False):
        mandatory = {
            'url': 'Source URL (-u)',
            'title': 'Story Title (--title)',
            'story': 'Content (-s)',
            'quote': 'Thumbnail Quote (-q)',
            'tags': 'Metadata Tags (-t)'
        }
        for attr, label in mandatory.items():
            if not str(getattr(args, attr, "")).strip():
                TerminalUI.notify(f"{TerminalUI.SYM_ERR} Missing Argument: {label}", TerminalUI.RED)
                sys.exit(1)

        if not re.match(reddit_regex, args.url):
            TerminalUI.notify(f"{TerminalUI.SYM_ERR} Invalid Format: --url must be a valid Reddit post link.", TerminalUI.RED)
            sys.exit(1)

        tags_list = [t.strip() for t in args.tags.split(',') if t.strip()]
        if len(tags_list) < 4:
            TerminalUI.notify(f"{TerminalUI.SYM_ERR} Insufficient Metadata: Minimum 4 tags required (found {len(tags_list)}).", TerminalUI.RED)
            sys.exit(1)

        db = load_database()
        for entry in db.get("stories", []):
            if entry.get("url") == args.url:
                TerminalUI.notify(f"{TerminalUI.SYM_ERR} Duplicate Found: URL already exists in database.", TerminalUI.RED)
                sys.exit(1)
            if entry.get("title") == args.title:
                TerminalUI.notify(f"{TerminalUI.SYM_ERR} Duplicate Found: Title already exists in database.", TerminalUI.RED)
                sys.exit(1)

    if getattr(args, 'fetch', None):
        if not re.match(reddit_regex, args.fetch):
            TerminalUI.notify(f"{TerminalUI.SYM_ERR} Invalid Link: --fetch requires a valid Reddit post URL.", TerminalUI.RED)
            sys.exit(1)

    if getattr(args, 'dl_video', False):
        if not str(getattr(args, 'url', "")).strip():
            TerminalUI.notify(f"{TerminalUI.SYM_ERR} Missing Argument: -u <url> is required for video downloads.", TerminalUI.RED)
            sys.exit(1)


def get_probe_data(path: str) -> Dict[str, Any]:
    """
    Extract technical metadata from a media file.
    
    Args:
        path: Path to the media file.
        
    Returns:
        JSON metadata extracted by ffprobe.
    """
    cmd = ['ffprobe', '-v', 'error', '-print_format', 'json', '-show_format', '-show_streams', path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return json.loads(result.stdout) if result.returncode == 0 else {}

def verify_output_integrity(path: str) -> Tuple[bool, str]:
    """
    Assess the validity and health of a generated media file.
    
    Args:
        path: Path to the file to verify.
        
    Returns:
        A tuple of (success_boolean, status_message).
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False, "File is missing or empty."
    try:
        metadata = get_probe_data(path)
        duration = float(metadata.get('format', {}).get('duration', 0))
        if duration < 3:
            return False, "Duration too short (minimum 3s required)."
        return True, f"Integrity Verified ({duration}s)"
    except Exception:
        return False, "Metadata extraction failed."


def fetch_content_payload(url: str) -> Optional[Dict[str, str]]:
    """
    Retrieve story content via the Reddit JSON endpoint.
    
    Args:
        url: The Reddit post URL.
        
    Returns:
        Dictionary containing the extracted title and selftext.
    """
    if requests is None:
        TerminalUI.notify("Error: 'requests' module unavailable.", TerminalUI.RED)
        return None

    TerminalUI.notify(f"Fetching Content: {url}", TerminalUI.CYAN)
    headers = {"User-Agent": USER_AGENT}
    
    try:
        target_url = url.rstrip("/") + ".json" if not url.endswith(".json") else url
        response = requests.get(target_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        raw_data = response.json()
        post_data = raw_data[0]['data']['children'][0]['data']
        return {"title": post_data['title'], "text": post_data['selftext']}
    except Exception as e:
        TerminalUI.notify(f"Fetch Failed: {e}", TerminalUI.RED)
        return None


def align_captions(audio_path: str, original_text: str) -> List[Dict[str, Any]]:
    """
    Synchronize recognized audio segments with original text tokens.
    
    Args:
        audio_path: Path to the generated narration audio.
        original_text: The source story content.
        
    Returns:
        List of dictionaries containing accurately timed word segments.
    """
    if pipeline is None:
        TerminalUI.notify("Error: transformers library not found.", TerminalUI.RED)
        return []

    TerminalUI.notify("Aligning captions with Whisper...", TerminalUI.CYAN)
    from difflib import SequenceMatcher

    pipe = pipeline("automatic-speech-recognition", model="openai/whisper-base.en", device=-1)
    result = pipe(audio_path, return_timestamps="word")
    chunks = result.get('chunks', [])
    
    fixed = []
    for c in chunks:
        t = c['text'].strip().lower()
        if fixed and (t.startswith("'") or t in ["s", "t", "re", "ve", "ll", "d", "m"]):
            fixed[-1]['text'] += c['text']
            fixed[-1]['timestamp'] = (fixed[-1]['timestamp'][0], c['timestamp'][1])
        else:
            fixed.append(c)
    chunks = fixed

    def tokenize(text):
        pattern = r"(?:[^\s.,!?;:]|(?<=\w)[.](?=\w))+[.,!?;:]*"
        tokens = re.findall(pattern, text)
        return [t.strip() for t in tokens if t.strip()]

    orig_tokens = tokenize(original_text)
    whisper_tokens = [c['text'].strip() for c in chunks]
    whisper_clean = [re.sub(r"[^\w'\"]", '', t).lower() for t in whisper_tokens]
    orig_clean = [re.sub(r"[^\w'\"]", '', t).lower() for t in orig_tokens]

    sm = SequenceMatcher(None, orig_clean, whisper_clean)
    aligned = [None] * len(orig_tokens)
    
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            for i, j in zip(range(i1, i2), range(j1, j2)):
                aligned[i] = chunks[j]['timestamp']

    last_end = 0.0
    for i in range(len(aligned)):
        if aligned[i] is None:
            next_start = None
            for j in range(i + 1, len(aligned)):
                if aligned[j]:
                    next_start = aligned[j][0]
                    break
            
            if next_start is None or next_start <= last_end:
                start = last_end
                end = last_end + 0.1
            else:
                gap_size = (next_start - last_end)
                missing_count = 0
                for j in range(i, len(aligned)):
                    if aligned[j] is None:
                        missing_count += 1
                    else:
                        break
                
                step = gap_size / (missing_count + 1)
                start = last_end + step
                end = start + step
            
            aligned[i] = (start, end)
        
        s, e = aligned[i]
        s = max(s, last_end)
        e = max(e, s + 0.03)
        
        aligned[i] = (s, e)
        last_end = e

    return [{'text': orig_tokens[i], 'timestamp': aligned[i]} for i in range(len(orig_tokens))]


def hex_to_ass_color(hex_color: str, alpha_percent: float = 1.0) -> str:
    """
    Convert a hex color code to the ASS subtitle color format.
    
    Args:
        hex_color: Hexadecimal color string (#RRGGBB).
        alpha_percent: Opacity as a decimal (1.0 = opaque, 0.0 = transparent).
        
    Returns:
        Formatted ASS color string (&HAABBGGRR).
    """
    try:
        c = str(hex_color).strip().lstrip('#')
        if len(c) == 6:
            r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
            alpha_val = int((1.0 - max(0.0, min(1.0, alpha_percent))) * 255)
            return f"&H{alpha_val:02X}{b:02X}{g:02X}{r:02X}"
    except (ValueError, TypeError):
        pass
    return "&H00FFFFFF"

def darken_color(hex_color: str, factor: float = 0.5) -> str:
    """
    Apply a darkening transformation to a hex color.
    
    Args:
        hex_color: Source color in hex format (#RRGGBB).
        factor: Darkening intensity (0.0 = black, 1.0 = original).
        
    Returns:
        The resulting darkened hex color.
    """
    try:
        c = str(hex_color).strip().lstrip('#')
        if len(c) == 6:
            r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
            return f"#{int(r*factor):02X}{int(g*factor):02X}{int(b*factor):02X}"
    except Exception:
        pass
    return hex_color

def generate_ass_subtitles(word_chunks: List[Dict], output_path: str, aspect_ratio: str = "16:9") -> None:
    """
    Synthesize an ASS subtitle file with dynamic highlighting and animation.
    
    Args:
        word_chunks: List of dictionaries containing timed word segments.
        output_path: Target filesystem path for the ASS file.
        aspect_ratio: The video frame's aspect ratio.
    """
    config = get_config()
    max_words = config.get("caption_max_words_per_line", 3)
    highlight_hex = config.get("caption_highlight_color", "#f8d568")
    highlight_color = hex_to_ass_color(highlight_hex)
    darken_factor = float(config.get("caption_glow_darken_factor", 0.7))
    dark_highlight_color = hex_to_ass_color(darken_color(highlight_hex, darken_factor))
    
    merged_chunks = []
    for item in word_chunks:
        text = item['text']
        if re.match(r"^[.,!?;:—\"\"\"'']+$", text) and merged_chunks:
            merged_chunks[-1]['text'] += text
            merged_chunks[-1]['timestamp'] = (merged_chunks[-1]['timestamp'][0], item['timestamp'][1])
        else:
            merged_chunks.append(item)
    
    captions = []
    idx = 0
    while idx < len(merged_chunks):
        best_j = idx + 1
        min_cost = float('inf')
        look_limit = min(idx + max_words + 2, len(merged_chunks))
        
        for j in range(idx + 1, look_limit + 1):
            current_slice = merged_chunks[idx:j]
            word_count = len(current_slice)
            cost = 0
            
            if word_count > max_words:
                cost += CAPTION_COST_WEIGHTS["overflow_penalty"] * (word_count - max_words)
            
            last_tok = current_slice[-1]['text']
            if any(p in last_tok for p in [".", "?", "!", ":"]):
                cost -= CAPTION_COST_WEIGHTS["period_bonus"]
            elif any(p in last_tok for p in [",", ";", "—"]):
                cost -= CAPTION_COST_WEIGHTS["comma_bonus"]
            
            cost += abs(word_count - max_words) * CAPTION_COST_WEIGHTS["length_deviation"]
            
            if j < len(merged_chunks):
                nxt = merged_chunks[j]['text']
                
                # Penalize bridging long gaps (pauses)
                if j > idx:
                    prev_end = merged_chunks[j-1]['timestamp'][1]
                    curr_start = merged_chunks[j]['timestamp'][0]
                    if (curr_start - prev_end) > 0.6:
                        cost += CAPTION_COST_WEIGHTS["gap_penalty"]

                if last_tok[0].isupper() and nxt[0].isupper():
                    cost += CAPTION_COST_WEIGHTS["capitalization_penalty"]

            if cost <= min_cost:
                min_cost = cost
                best_j = j
        
        captions.append(merged_chunks[idx:best_j])
        idx = best_j

    base_size = int(config.get("caption_font_size", 100))
    fs = base_size if aspect_ratio == "9:16" else int(base_size * 0.75)
    
    quality_map = {"1080p": 1.0, "2048p": 2048 / 1920, "4096p": 4096 / 1920}
    scale = quality_map.get(config.get("quality", "1080p"), 1.0)
    res_x, res_y = (int(1080 * scale), int(1920 * scale)) if aspect_ratio == "9:16" else (int(1920 * scale), int(1080 * scale))
    
    font = config.get('caption_font', 'Arial Black')
    primary_color = hex_to_ass_color(config.get("caption_font_color", "#FFFFFF"))
    
    shadow_enabled = config.get("caption_font_shadow", True)
    shadow_color = hex_to_ass_color(
        darken_color(config.get("caption_shadow_color", "#FFFFFF"), darken_factor), 
        float(str(config.get("caption_strength", "40%")).replace("%", "")) / 100.0
    )
    shadow_size = 2 if shadow_enabled else 0
    
    glow_enabled = config.get("caption_glow", True)
    glow_color = hex_to_ass_color(
        darken_color(config.get("caption_glow_color", "#FFFFFF"), darken_factor),
        float(str(config.get("caption_glow_strength", "75%")).replace("%", "")) / 100.0
    )
    outline_size = 2 if glow_enabled else 0
    
    align_val = 5 if config.get("caption_position", "center") == "center" else 2
    margin_v = 50 if align_val == 2 else 0

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {res_x}
PlayResY: {res_y}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{fs},{primary_color},&H000000FF,{glow_color},{shadow_color},-1,0,0,0,100,100,1,0,1,{outline_size},{shadow_size},{align_val},20,20,{margin_v},1
"""
    events = "\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    
    def ts(s):
        return f"{int(s // 3600)}:{int((s % 3600) // 60):02d}:{s % 60:05.2f}"

    TerminalUI.notify("Generating caption events...", TerminalUI.CYAN)
    
    for c_idx, chunk in enumerate(captions):
        for a_idx, a_word in enumerate(chunk):
            s_t = a_word['timestamp'][0]
            
            if a_idx < len(chunk) - 1:
                e_t = chunk[a_idx+1]['timestamp'][0]
            elif c_idx < len(captions) - 1:
                e_t = captions[c_idx+1][0]['timestamp'][0]
            else:
                e_t = chunk[-1]['timestamp'][1]
            
            line_parts = []
            for j, word in enumerate(chunk):
                wt = word['text'].upper().replace('"', '\\"').replace("'", "'")
                if j == a_idx:
                    pop = r"{\t(0,80,\fscx108\fscy108)\t(80,160,\fscx100\fscy100)}"
                    line_parts.append(rf"{pop}{{\c{highlight_color}&\3c{dark_highlight_color}&\4c{dark_highlight_color}&}}{wt}{{\c{primary_color}&\3c{glow_color}&\4c{shadow_color}&}}")
                else:
                    line_parts.append(wt)
            
            final_text = r"{\blur2\q2}" + " ".join(line_parts)
            events += f"Dialogue: 0,{ts(s_t)},{ts(e_t)},Default,,0,0,0,,{final_text}\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header + events)


def generate_vocal_audio(text_content: str, destination_name: str) -> str:
    """
    Produce narration audio with multi-layer fallback strategy.
    
    Args:
        text_content: Plaintext content to narrate.
        destination_name: Base filename (slug) for the audio assets.
        
    Returns:
        Absolute path to the finalized audio file.
    """
    api_key_pool = os.getenv("ELEVENLABS_API_KEYS", "").split(",")
    config = get_config()
    voice_id = config.get("voice_id", "pNInz6obpgDQGcFmaJgB")
    
    model_id = "eleven_multilingual_v2"
    if config.get("voice_model") == "v3":
        model_id = "eleven_multilingual_v3"
        
    final_output = str(Paths['tts_generations'] / f"{destination_name}.mp3")

    for index, key in enumerate(api_key_pool):
        clean_key = key.strip()
        if not clean_key or clean_key == "placeholder":
            continue

        TerminalUI.notify(f"Attempting ElevenLabs (Account {index+1})...", TerminalUI.CYAN)
        command = [
            "curl", "-s", "-X", "POST",
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            "-H", f"xi-api-key: {clean_key}",
            "-H", "Content-Type: application/json",
            "-d", json.dumps({"text": text_content, "model_id": model_id}),
            "--output", final_output
        ]
        
        if subprocess.run(command).returncode == 0 and os.path.exists(final_output):
            with open(final_output, 'rb') as f:
                header_bytes = f.read(4)
                if header_bytes.startswith(b'ID3') or header_bytes.startswith(b'\xff\xfb') or header_bytes.startswith(b'\xff\xf3'):
                    return final_output
                else:
                    error_payload = f.read(512).decode('utf-8', errors='ignore')
                    if index == len(api_key_pool) - 1 or "quota" not in error_payload.lower():
                        TerminalUI.notify(f"ElevenLabs Error (Key {index+1}): {error_payload[:80]}...", TerminalUI.YELLOW)
            
            if os.path.exists(final_output):
                os.remove(final_output)
        time.sleep(1)

    TerminalUI.notify("Using local TTS fallback...", TerminalUI.CYAN)
    local_output = str(Paths['tts_generations'] / f"{destination_name}.wav")
    return generate_local_audio(text_content, local_output)


def parse_hex_color(hex_str: str, fallback: Tuple[int, int, int] = (255, 255, 255)) -> Tuple[int, int, int]:
    """
    Parse a hexadecimal color string into an RGB integer tuple.
    
    Args:
        hex_str: The hex color string (#RRGGBB).
        fallback: RGB tuple to return on parsing failure.
        
    Returns:
        A tuple of (Red, Green, Blue) integers.
    """
    try:
        clean = str(hex_str).strip().lstrip('#')
        if len(clean) == 6:
            return tuple(int(clean[i:i+2], 16) for i in (0, 2, 4))
    except (ValueError, TypeError):
        pass
    return fallback

def resolve_system_font(name: str, size: int) -> ImageFont.FreeTypeFont:
    """
    Locate a system font by name with cross-platform fallback logic.
    
    Args:
        name: The requested font family name.
        size: The desired font point size.
        
    Returns:
        A loaded ImageFont object.
    """
    try:
        return ImageFont.truetype(name, size)
    except Exception:
        pass

    if sys.platform.startswith("linux") and shutil.which("fc-match"):
        try:
            result = subprocess.run(["fc-match", "-f", "%{file}", name], capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                font_file = result.stdout.strip()
                if os.path.exists(font_file):
                    return ImageFont.truetype(font_file, size)
        except Exception:
            pass
            
    return ImageFont.load_default()

def resolve_emoji_font(size: int) -> Optional[ImageFont.FreeTypeFont]:
    """Locate a color emoji font on the system."""
    emoji_fonts = ["Noto Color Emoji", "Apple Color Emoji", "Segoe UI Emoji"]
    for name in emoji_fonts:
        # Try requested size first, then common bitmap sizes (109)
        for target_size in [size, 109, 128]:
            try:
                return ImageFont.truetype(name, target_size)
            except Exception:
                pass
        
        # Linux fontconfig fallback
        if sys.platform.startswith("linux") and shutil.which("fc-match"):
            try:
                result = subprocess.run(["fc-match", "-f", "%{file}", name], capture_output=True, text=True)
                if result.returncode == 0 and result.stdout.strip():
                    font_file = result.stdout.strip()
                    if os.path.exists(font_file):
                        for target_size in [size, 109, 128]:
                            try:
                                return ImageFont.truetype(font_file, target_size)
                            except Exception:
                                pass
            except Exception:
                pass
    return None


class ThumbnailGenerator:
    """Orchestrates the creation of cinematic preview images for generated videos."""
    
    @staticmethod
    def create(video_src: str, text_overlay: str, output_dst: str, seek_time: float = 0.0) -> None:
        """
        Produce a blurred-background thumbnail with stylized text overlay.
        
        Args:
            video_src: Path to the source MP4 file.
            text_overlay: The title or quote to display on the image.
            output_dst: Target filesystem path for the resulting PNG.
            seek_time: Time offset in seconds to capture the frame from.
        """
        if Image is None:
            TerminalUI.notify("Pillow not found, thumbnail generation skipped.", TerminalUI.YELLOW)
            return

        TerminalUI.notify("Generating thumbnail...", TerminalUI.CYAN)
        config = get_config()
        working_frame = str(Paths['data'] / "frame_capture.jpg")

        try:
            video_meta = get_probe_data(video_src)
            total_dur = float(video_meta.get('format', {}).get('duration', 5))
            
            # Use provided seek_time, but ensure it's valid
            capture_point = seek_time
            if capture_point <= 0 or capture_point >= total_dur:
                 capture_point = random.uniform(
                    total_dur * VIDEO_CAPTURE_RANGE[0],
                    total_dur * VIDEO_CAPTURE_RANGE[1]
                )
            
            subprocess.run(
                ["ffmpeg", "-y", "-ss", str(capture_point), "-i", video_src, "-frames:v", "1", working_frame],
                capture_output=True
            )

            if not os.path.exists(working_frame):
                raise FileNotFoundError("FFmpeg frame extraction failed.")

            with Image.open(working_frame) as base_img:
                blur_factor = float(str(config.get("thumbnail_bg_blur", "50%")).replace("%", ""))
                canvas = base_img.filter(
                    ImageFilter.GaussianBlur(radius=(blur_factor / 100) * GAUSSIAN_BLUR_MULTIPLIER)
                )
                wid, hgt = canvas.size

                fs = int(config.get("thumbnail_font_size", 100))
                typeface = resolve_system_font(config.get("thumbnail_font", "Arial Black"), fs)
                
                darken_factor = float(config.get("caption_glow_darken_factor", 0.7))
                txt_color = parse_hex_color(config.get("thumbnail_font_color", "#FFFFFF"))
                sh_color = parse_hex_color(darken_color(config.get("thumbnail_shadow_color", "#000000"), darken_factor))
                
                sh_strength = float(str(config.get("thumbnail_strength", "20%")).replace("%", ""))
                sh_alpha = int((sh_strength / 100) * 255)
                sh_fill = (*sh_color, sh_alpha)

                layer = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(layer)

                line_buffer, word_pool = [], text_overlay.split()
                current_line = ""
                for word in word_pool:
                    if len(current_line + " " + word) < THUMBNAIL_CONFIG["max_chars_per_line"]:
                        current_line += " " + word
                    else:
                        line_buffer.append(current_line.strip())
                        current_line = word
                line_buffer.append(current_line.strip())
                line_buffer = [l for l in line_buffer if l]

                spacing = int(fs * THUMBNAIL_CONFIG["line_spacing_multiplier"])
                y_cursor = (hgt - (len(line_buffer) * spacing)) / 2

                emo_font = resolve_emoji_font(fs)

                def draw_complex_text(draw_obj, pos, text, main_font, emoji_font, color, use_embedded_color=True):
                    x, y = pos
                    std_bbox = draw_obj.textbbox((0, 0), "Hg", font=main_font)
                    line_h = std_bbox[3] - std_bbox[1]
                    
                    for char in text:
                        is_emoji = ord(char) > 0x2000
                        active_font = emoji_font if (is_emoji and emoji_font) else main_font
                        
                        if is_emoji and emoji_font:
                            # Render emoji to separate layer and scale for bitmap support
                            emo_size = emoji_font.size
                            char_bbox = draw_obj.textbbox((0, 0), char, font=active_font)
                            char_w = char_bbox[2] - char_bbox[0]
                            char_h = char_bbox[3] - char_bbox[1]
                            
                            # Create a patch at native font size
                            patch = Image.new('RGBA', (max(1, char_w), max(1, char_h)), (0, 0, 0, 0))
                            p_draw = ImageDraw.Draw(patch)
                            p_draw.text((0, 0), char, font=active_font, fill=color, embedded_color=use_embedded_color)
                            
                            # Scale to match line height
                            scale = line_h / char_h
                            new_size = (int(char_w * scale), int(line_h))
                            patch = patch.resize(new_size, Image.Resampling.LANCZOS)
                            
                            # Composite onto canvas
                            offset_y = (line_h - new_size[1]) // 2
                            canvas.paste(patch, (int(x), int(y + offset_y)), patch)
                            x += new_size[0]
                        else:
                            char_bbox = draw_obj.textbbox((0, 0), char, font=active_font)
                            char_h = char_bbox[3] - char_bbox[1]
                            offset_y = (line_h - char_h) // 2
                            
                            draw_obj.text((x, y + offset_y), char, font=active_font, fill=color)
                            
                            bbox = draw_obj.textbbox((x, y), char, font=active_font)
                            x += (bbox[2] - bbox[0])

                if config.get("thumbnail_glow", True):
                    glow_color = parse_hex_color(darken_color(config.get("thumbnail_glow_color", "#FFFFFF"), darken_factor))
                    glow_strength = float(str(config.get("thumbnail_glow_strength", "75%")).replace("%", "")) / 100.0
                    glow_alpha = int(glow_strength * 255)
                    
                    glow_layer = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
                    g_draw = ImageDraw.Draw(glow_layer)
                    
                    gy_cursor = (hgt - (len(line_buffer) * spacing)) / 2
                    for content in line_buffer:
                        box = draw.textbbox((0, 0), content, font=typeface)
                        tw = box[2] - box[0]
                        gx_cursor = (wid - tw) / 2
                        
                        for ox in range(-2, 3):
                            for oy in range(-2, 3):
                                draw_complex_text(g_draw, (gx_cursor + ox, gy_cursor + oy), content, typeface, emo_font, (*glow_color, glow_alpha), use_embedded_color=False)
                        gy_cursor += spacing
                        
                    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(10))
                    layer.alpha_composite(glow_layer)



                for content in line_buffer:
                    box = draw.textbbox((0, 0), content, font=typeface)
                    tw = box[2] - box[0]
                    x_cursor = (wid - tw) / 2

                    emo_font = resolve_emoji_font(fs)

                    if config.get("thumbnail_font_shadow", True):
                        drift = max(2, fs // THUMBNAIL_CONFIG["shadow_offset_divisor"])
                        draw_complex_text(draw, (x_cursor + drift, y_cursor + drift), content, typeface, emo_font, sh_fill, use_embedded_color=False)

                    draw_complex_text(draw, (x_cursor, y_cursor), content, typeface, emo_font, (*txt_color, 255), use_embedded_color=True)
                    y_cursor += spacing

                layer = layer.rotate(
                    random.uniform(*THUMBNAIL_CONFIG["rotation_range"]),
                    resample=Image.BICUBIC
                )
                canvas.paste(layer, (0, 0), layer)
                canvas.save(output_dst, "PNG")

            if os.path.exists(working_frame):
                os.remove(working_frame)
        except Exception as e:
            TerminalUI.notify(f"Thumbnail generation failed: {e}", TerminalUI.RED)

def find_system_ffmpeg() -> str:
    """
    Locate an FFmpeg binary equipped with libass support.
    
    Returns:
        The filesystem path to the valid FFmpeg executable.
    """
    search_paths = [
        shutil.which("ffmpeg"),
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/opt/homebrew/bin/ffmpeg",
        "/bin/ffmpeg"
    ]
    
    unique_paths = []
    seen = set()
    for entry in search_paths:
        if entry and entry not in seen and os.path.exists(entry):
            unique_paths.append(entry)
            seen.add(entry)
            
    for entry in unique_paths:
        try:
            probe = subprocess.run([entry, "-version"], capture_output=True, text=True, check=False)
            if "--enable-libass" in probe.stdout:
                return entry
        except Exception:
            continue
            
    return unique_paths[0] if unique_paths else "ffmpeg"

def compose_final_video(bg_src: str, audio_src: str, subtitle_src: str, output_dst: str, aspect_ratio: str = "16:9", start_offset: float = 0.0) -> None:
    """
    Orchestrate video and audio stream merger via FFmpeg filters.
    
    Args:
        bg_src: Path to the background MP4 file.
        audio_src: Path to the narration MP3/WAV file.
        subtitle_src: Path to the generated ASS file.
        output_dst: Final target path for the production video.
        aspect_ratio: Target display format (e.g., '9:16').
        start_offset: Time in seconds to start reading the background video.
    """
    TerminalUI.notify("Composing final video...", TerminalUI.CYAN)
    config = get_config()
    fitting_mode = config.get("background_fit", "crop")
    
    audio_meta = get_probe_data(audio_src)
    audio_dur = float(audio_meta.get('format', {}).get('duration', 0))
    
    video_meta = get_probe_data(bg_src)
    video_dur = float(video_meta.get('format', {}).get('duration', 0))

    audio_dur = float(audio_meta.get('format', {}).get('duration', 0))
    
    # Use pre-calculated offset
    pass

    qualities = {"1080p": 1.0, "2048p": 2048 / 1920, "4096p": 4096 / 1920}
    scale_factor = qualities.get(config.get("quality", "1080p"), 1.0)
    target_w, target_h = (int(1080 * scale_factor), int(1920 * scale_factor)) if aspect_ratio == "9:16" else (int(1920 * scale_factor), int(1080 * scale_factor))

    if aspect_ratio == "9:16" and fitting_mode == "pad":
        v_filter = f"split[a][b];[a]scale={target_w}:{target_h}:force_original_aspect_ratio=increase,crop={target_w}:{target_h},boxblur=20:10[bg];[b]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2"
    elif fitting_mode == "crop":
        v_filter = f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase,crop={target_w}:{target_h}"
    else:
        v_filter = f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2"

    if config.get("captions", True) and os.path.exists(subtitle_src):
        escaped_ass = str(subtitle_src).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
        v_filter += f",subtitles=f='{escaped_ass}'"

    bg_vol = float(str(config.get("background_volume", "0%")).replace("%", "")) / 100.0
    narr_vol = float(str(config.get("narration_volume", "100%")).replace("%", "")) / 100.0
    
    if bg_vol <= 0.01:
        pipeline_args = ["-filter_complex", f"[0:v]{v_filter}[outv];[1:a]volume={narr_vol}[outa]", "-map", "[outv]", "-map", "[outa]"]
    else:
        audio_graph = f"[0:a]volume={bg_vol}[bga];[1:a]volume={narr_vol}[ttsa];[bga][ttsa]amix=inputs=2:duration=shortest[outa]"
        pipeline_args = ["-filter_complex", f"[0:v]{v_filter}[outv];{audio_graph}", "-map", "[outv]", "-map", "[outa]"]

    ffmpeg_bin = find_system_ffmpeg()
    execution_cmd = [
        ffmpeg_bin, "-y", "-ss", f"{start_offset:.2f}", "-i", bg_src, "-i", audio_src,
        "-t", f"{audio_dur:.2f}"
    ] + pipeline_args + ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-shortest", output_dst]
    
    try:
        subprocess.run(execution_cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        TerminalUI.notify(f"FFmpeg composition failed: {e.stderr.decode()}", TerminalUI.RED)
        raise


def execute_generation_pipeline(url: str, title: str, content: str, aspect_ratio: str, quote: str = "", tags: str = "") -> None:
    """
    Execute the full end-to-end video production workflow.
    
    Args:
        url: Source content URL.
        title: Video title.
        content: The story text.
        aspect_ratio: Target display format.
        quote: Optional thumbnail quote.
        tags: Comma-separated metadata tags.
    """
    safe_slug = "".join(x for x in title if x.isalnum() or x in " -_").strip().replace(" ","_")
    destination_dir = Paths['output'] / safe_slug

    if destination_dir.exists():
         TerminalUI.notify(f"{TerminalUI.SYM_ERR} Directory already exists: {destination_dir}", TerminalUI.RED)
         return

    available_bgs = list(Paths['background_videos'].glob("*.mp4"))
    if not available_bgs:
        TerminalUI.notify(f"{TerminalUI.SYM_ERR} No background videos found in {Paths['background_videos']}", TerminalUI.RED)
        TerminalUI.notify("Tip: Run 'omni-stories --doctor' to download sample assets.", TerminalUI.YELLOW)
        return

    chosen_bg = str(random.choice(available_bgs))
    destination_dir.mkdir(parents=True, exist_ok=True)
    
    TerminalUI.notify(f"Starting production: {TerminalUI.BOLD}{title}{TerminalUI.NC}", TerminalUI.CYAN)
    
    audio_path = generate_vocal_audio(content, safe_slug)
    aligned_words = align_captions(audio_path, content)
    
    ass_path = Paths['tts_generations'] / f"{safe_slug}.ass"
    generate_ass_subtitles(aligned_words, str(ass_path), aspect_ratio)
    
    video_meta = get_probe_data(chosen_bg)
    video_dur = float(video_meta.get('format', {}).get('duration', 0))
    audio_meta = get_probe_data(audio_path)
    audio_dur = float(audio_meta.get('format', {}).get('duration', 0))

    start_offset = 0.0
    if video_dur > audio_dur + 2:
        start_offset = random.uniform(0, video_dur - audio_dur - 2)

    video_path = str(destination_dir / f"{safe_slug}.mp4")
    compose_final_video(chosen_bg, audio_path, str(ass_path), video_path, aspect_ratio, start_offset)
    
    # Use raw background and exact offset for clean thumbnail
    ThumbnailGenerator.create(chosen_bg, quote or title, str(destination_dir / "thumbnail.png"), start_offset + 1.0)
    
    with open(destination_dir / "metadata.md", 'w') as f:
        f.write(f"Title: {title}\n")
        f.write(f"URL: {url}\n")
        f.write(f"Date: {time.ctime()}\n\n")
        f.write(f"{content}")

    success, message = verify_output_integrity(video_path)
    if success:
        db = load_database()
        tag_entities = [t.strip() for t in tags.split(',') if t.strip()]
        db['stories'].append({
            "title": title,
            "url": url,
            "tags": tag_entities,
            "timestamp": time.ctime()
        })
        save_database(db)
            
        TerminalUI.notify(f"{TerminalUI.SYM_OK} {message}")
        TerminalUI.notify(f"{TerminalUI.SYM_OK} Generation complete: {destination_dir}", TerminalUI.GREEN)
    else:
        TerminalUI.notify(f"{TerminalUI.SYM_ERR} Generation failed: {message}", TerminalUI.RED)


def verify_python_environment() -> List[str]:
    """Identify missing Python packages from requirements"""
    dependencies = {
        'yaml': 'pyyaml',
        'dotenv': 'python-dotenv',
        'numpy': 'numpy',
        'transformers': 'transformers',
        'torch': 'torch',
        'requests': 'requests',
        'yt_dlp': 'yt-dlp',
        'phonemizer': 'phonemizer',
        'onnxruntime': 'onnxruntime',
        'soundfile': 'soundfile',
        'PIL': 'Pillow'
    }
    missing = []
    for module, package in dependencies.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    return missing


def run_setup_routine(api_key: Optional[str] = None) -> None:
    """
    Configure the application environment and install dependencies.
    
    Args:
        api_key: Optional ElevenLabs API key for initial configuration.
    """
    TerminalUI.print_large_banner()
    TerminalUI.notify("Initializing Omni-Stories Environment...", TerminalUI.CYAN)
    
    if api_key and api_key != "NO_KEY":
        set_key(ENV_PATH, "ELEVENLABS_API_KEYS", api_key.strip())
        load_dotenv(ENV_PATH, override=True)
        TerminalUI.notify(f"{TerminalUI.SYM_OK} Credentials configured.")
    
    TerminalUI.notify("Checking dependencies...", TerminalUI.CYAN)
    missing = verify_python_environment()
    if missing:
        TerminalUI.notify(f"Installing missing packages: {', '.join(missing)}", TerminalUI.YELLOW)
        pip_cmd = [sys.executable, "-m", "pip", "install", "-r", str(ENGINE_DIR / "requirements.txt")]
        
        if sys.prefix == sys.base_prefix:
             TerminalUI.notify("System-wide Python detected, using user-space installation.", TerminalUI.MAGENTA)
             pip_cmd.append("--user")
             if sys.version_info >= (3, 11):
                 pip_cmd.append("--break-system-packages")

        try:
            subprocess.check_call(pip_cmd)
            TerminalUI.notify(f"{TerminalUI.SYM_OK} Dependencies successfully installed.")
        except subprocess.CalledProcessError as e:
            TerminalUI.notify(f"{TerminalUI.SYM_ERR} Dependency installation failed: {e}", TerminalUI.RED)
            sys.exit(1)
    else:
        TerminalUI.notify(f"{TerminalUI.SYM_OK} Project dependencies are satisfied.")

    binary_dir = Path.home() / ".local" / "bin"
    binary_dir.mkdir(parents=True, exist_ok=True)
    executable_link = binary_dir / "omni-stories"
    
    try:
        if executable_link.exists() or executable_link.is_symlink():
            executable_link.unlink()

        wrapper = f'#!/bin/bash\nexec "{sys.executable}" "{ENGINE_DIR}/main.py" "$@"\n'
        with open(executable_link, "w") as f:
            f.write(wrapper)
        executable_link.chmod(0o755)
        TerminalUI.notify(f"{TerminalUI.SYM_OK} Global executable established: {executable_link}")
    except OSError as e:
        TerminalUI.notify(f"{TerminalUI.SYM_ERR} Failed to create executable link: {e}", TerminalUI.RED)
    
    for folder in [Paths['data'], Paths['output'], Paths['background_videos'], Paths['models'], Paths['tts_generations']]:
        folder.mkdir(parents=True, exist_ok=True)
        
    TerminalUI.notify(f"\n{TerminalUI.GREEN}Environment Setup Complete.{TerminalUI.NC}")


def run_health_check() -> None:
    """
    Perform a comprehensive system diagnostic with automated self-healing.
    Verifies dependencies, API keys, model files, and directory structure.
    """
    TerminalUI.print_large_banner()
    TerminalUI.notify("Initiating System Diagnostic...", TerminalUI.CYAN)
    
    missing_deps = verify_python_environment()
    if missing_deps:
        TerminalUI.notify(f"{TerminalUI.SYM_WARN} Missing Python packages: {', '.join(missing_deps)}", TerminalUI.YELLOW)
        if TerminalUI.input_prompt("Attempt automated installation? (y/N)").lower() == "y":
            run_setup_routine(api_key="NO_KEY")
            return

    for key, path in Paths.items():
        if key != 'database' and not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            TerminalUI.notify(f"Healed missing directory: {path}", TerminalUI.YELLOW)
    
    binaries = {"ffmpeg": "Video composition", "ffprobe": "Metadata extraction"}
    for tool, desc in binaries.items():
        if not shutil.which(tool):
             TerminalUI.notify(f"{TerminalUI.SYM_ERR} Binary Missing: {tool} ({desc})", TerminalUI.RED)
        else:
            TerminalUI.notify(f"{TerminalUI.SYM_OK} System Binary Found: {tool}")
            if tool == "ffmpeg":
                try:
                    res = subprocess.run([tool, "-version"], capture_output=True, text=True)
                    if "--enable-libass" in res.stdout:
                        TerminalUI.notify(f"{TerminalUI.SYM_OK} FFmpeg: Subtitles Enabled (libass)")
                except Exception:
                    pass
    
    keys = os.getenv("ELEVENLABS_API_KEYS", "")
    if not keys or "placeholder" in keys:
        TerminalUI.notify(f"{TerminalUI.SYM_WARN} No ElevenLabs API keys configured in .env", TerminalUI.YELLOW)
    else:
        TerminalUI.notify(f"{TerminalUI.SYM_OK} API Credentials: Configured ({len(keys.split(','))} keys)")

    kokoro_model = Paths['kokoro'] / "model.onnx"
    if not kokoro_model.exists():
        TerminalUI.notify(f"{TerminalUI.SYM_WARN} Kokoro TTS model missing.", TerminalUI.YELLOW)
        if TerminalUI.input_prompt("Download sample Kokoro model? (y/N)").lower() == "y":
            TerminalUI.notify("Downloading placeholder model...", TerminalUI.CYAN)
            url = "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v0_19.onnx"
            try:
                subprocess.run(["curl", "-L", url, "--output", str(kokoro_model)], check=True)
                TerminalUI.notify(f"{TerminalUI.SYM_OK} Model downloaded.")
            except Exception as e:
                TerminalUI.notify(f"Download failed: {e}", TerminalUI.RED)

    bgs = list(Paths['background_videos'].glob("*.mp4"))
    if not bgs:
        TerminalUI.notify(f"{TerminalUI.SYM_WARN} No background footage available.", TerminalUI.YELLOW)
        if TerminalUI.input_prompt("Download sample background? (y/N)").lower() == "y":
            sample_url = "https://www.youtube.com/watch?v=GVgLf8ENBJo"
            TerminalUI.notify(f"Recommended tool: omni-stories --dl_video -u {sample_url}", TerminalUI.WHITE)

    db = load_database()
    TerminalUI.notify(f"{TerminalUI.SYM_OK} Database Health: Verified ({len(db.get('stories', []))} entries)")
    TerminalUI.notify(f"\n{TerminalUI.GREEN}Diagnostic Complete.{TerminalUI.NC}")


def display_help() -> None:
    """Print the command-line interface usage guide."""
    TerminalUI.print_banner()
    print(f"{TerminalUI.CYAN}Usage:{TerminalUI.NC} omni-stories [COMMAND]")
    print("")
    print(f"{TerminalUI.BOLD}Commands:{TerminalUI.NC}")
    print(f"  {TerminalUI.GREEN}--new{TerminalUI.NC} {TerminalUI.PINK}-u <url> -tt <title> -s <story> -q <quote> -t <tags>{TerminalUI.NC}")
    print(f"                              Synthesize a new production video.")
    print(f"  {TerminalUI.GREEN}--fetch{TerminalUI.NC} {TerminalUI.PINK}<url>{TerminalUI.NC}               Harvest story content from Reddit.")
    print(f"  {TerminalUI.GREEN}--dl_video{TerminalUI.NC} {TerminalUI.PINK}-u <url> -r <res>{TerminalUI.NC} Acquire background footage.")
    print(f"  {TerminalUI.GREEN}--list{TerminalUI.NC} {TerminalUI.PINK}[n]{TerminalUI.NC}                  List historical productions (default: 5).")
    print(f"  {TerminalUI.GREEN}--remove{TerminalUI.NC} {TerminalUI.PINK}<n|n-m>{TerminalUI.NC}          Delete database entries.")
    print(f"  {TerminalUI.GREEN}--set{TerminalUI.NC} {TerminalUI.PINK}<key1,key2,...>{TerminalUI.NC}    Update ElevenLabs API credentials.")
    print(f"  {TerminalUI.GREEN}--doctor{TerminalUI.NC}                    Perform system diagnostic.")
    print(f"  {TerminalUI.GREEN}--uninstall{TerminalUI.NC}                 Purge application data.")


class CLIParser(argparse.ArgumentParser):
    """Custom command-line argument parser with specialized error handling."""
    
    def error(self, message: str) -> None:
        """Handle and format parsing errors before exit."""
        TerminalUI.notify(f"Command Error: {message}", TerminalUI.RED)
        display_help()
        sys.exit(1)


def main() -> None:
    """Entry point for the application; routes CLI commands to internal modules."""
    parser = CLIParser(prog="omni-stories", add_help=False)
    
    parser.add_argument("--help", action="store_true")
    parser.add_argument("--new", action="store_true")
    parser.add_argument("-u", "--url")
    parser.add_argument("-tt", "--title")
    parser.add_argument("-s", "--story")
    parser.add_argument("-q", "--quote")
    parser.add_argument("-t", "--tags")
    
    parser.add_argument("--fetch", metavar="URL")
    parser.add_argument("--dl_video", action="store_true")
    parser.add_argument("-r", "--resolution", choices=["720p", "1080p", "1440p", "2160p"], default="1080p")
    parser.add_argument("--list", nargs="?", type=int, const=5)
    parser.add_argument("--remove")
    parser.add_argument("--set", nargs="+")
    parser.add_argument("--doctor", action="store_true")
    parser.add_argument("--uninstall", action="store_true")
    parser.add_argument("--install", nargs="?", const="NO_KEY")

    if len(sys.argv) == 1 or "--help" in sys.argv:
        display_help()
        return

    args = parser.parse_args()
    config = get_config()

    if args.install:
        run_setup_routine(args.install)
        return
        
    if args.doctor:
        run_health_check()
        return

    validate_cli_input(args, config)

    if args.new:
        execute_generation_pipeline(args.url, args.title, args.story, config.get("aspect_ratio", "16:9"), args.quote, args.tags)
    
    elif args.fetch:
        payload = fetch_content_payload(args.fetch)
        if payload:
            TerminalUI.notify(f"TITLE: {payload['title']}", TerminalUI.WHITE)
            TerminalUI.notify(f"CONTENT: {payload['text']}", TerminalUI.CYAN)

    elif args.dl_video:
        if yt_dlp is None:
            TerminalUI.notify("Error: 'yt-dlp' module not found.", TerminalUI.RED)
            return
        
        target_res = args.resolution or "1080p"
        height = {"720p": 720, "1080p": 1080, "1440p": 1440, "2160p": 2160}.get(target_res, 1080)
        
        TerminalUI.notify(f"Downloading background at {target_res}...", TerminalUI.CYAN)
        ydl_config = {
            'format': f"bestvideo[height<={height}][ext=mp4]+bestaudio[ext=m4a]/best[height<={height}][ext=mp4]",
            'outtmpl': str(Paths['background_videos'] / "%(title)s.%(ext)s"),
            'quiet': True,
            'no_warnings': True,
            'http_headers': {'User-Agent': USER_AGENT}
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_config) as ydl:
                ydl.download([args.url])
            TerminalUI.notify(f"{TerminalUI.SYM_OK} Download complete.")
        except Exception as e:
            TerminalUI.notify(f"Download failed: {e}", TerminalUI.RED)

    elif args.list:
        db = load_database()
        stories = db.get("stories", [])
        if not stories:
            TerminalUI.notify("Production database is currently empty.", TerminalUI.YELLOW)
        else:
            for i, s in enumerate(reversed(stories[-args.list:])):
                curr_idx = len(stories) - i
                TerminalUI.notify(f"{TerminalUI.PINK}{curr_idx}. {TerminalUI.WHITE}{s.get('title')}", TerminalUI.BOLD)
                TerminalUI.notify(f"   URL: {s.get('url')}", TerminalUI.CYAN)
                TerminalUI.notify(f"   Tags: {', '.join(s.get('tags', []))}", TerminalUI.CYAN)
                TerminalUI.notify("-" * 40, TerminalUI.BLUE)

    elif args.remove:
        db = load_database()
        entries = db.get("stories", [])
        try:
            if "-" in args.remove:
                start, end = map(int, args.remove.split("-"))
                indices = set(range(start, end + 1))
            else:
                indices = {int(args.remove)}
            
            db["stories"] = [s for i, s in enumerate(entries, 1) if i not in indices]
            save_database(db)
            TerminalUI.notify(f"{TerminalUI.SYM_OK} Removed {len(entries) - len(db['stories'])} database entries.")
        except ValueError:
            TerminalUI.notify("Invalid range format. Use N or N-M.", TerminalUI.RED)

    elif args.set:
        keys = "".join(args.set).replace(" ", "")
        set_key(ENV_PATH, "ELEVENLABS_API_KEYS", keys)
        TerminalUI.notify(f"{TerminalUI.SYM_OK} API credentials updated.")

    elif args.uninstall:
        uninstall_script = Paths['data'] / "uninstall.sh"
        if uninstall_script.exists():
            try:
                # Use subprocess to run the bash script directly
                subprocess.run(["bash", str(uninstall_script)], check=True)
            except subprocess.CalledProcessError:
                TerminalUI.notify("Uninstall script failed or was aborted.", TerminalUI.RED)
            except Exception as e:
                TerminalUI.notify(f"Error launching uninstaller: {e}", TerminalUI.RED)
        else:
            TerminalUI.notify("Uninstall script not found. Please remove data manually.", TerminalUI.RED)


if __name__ == "__main__":
    main()
