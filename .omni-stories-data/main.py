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


# Custom Exception Hierarchy for better error handling and debugging
class OmniStoriesError(Exception):
    """Base exception for all Omni-Stories errors"""
    pass

class ConfigError(OmniStoriesError):
    """Raised when configuration is invalid or missing"""
    pass

class ValidationError(OmniStoriesError):
    """Raised when input validation fails"""
    pass

class APIError(OmniStoriesError):
    """Raised when external API calls fail"""
    pass

class DependencyError(OmniStoriesError):
    """Raised when required dependencies are missing"""
    pass

class FileSystemError(OmniStoriesError):
    """Raised when file operations fail"""
    pass

class ModelError(OmniStoriesError):
    """Raised when ML model operations fail"""
    pass


VERSION = "1.1.0"

class TerminalUI:
    """ANSI-based terminal output formatting with integrated file logging."""
    
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
    _logger = None

    @classmethod
    def setup_logging(cls) -> None:
        """Initialize usage_logger with rotation."""
        import logging
        from logging.handlers import RotatingFileHandler
        
        if cls._logger:
            return

        log_dir = ENGINE_DIR / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / "omni-stories.log"
        
        cls._logger = logging.getLogger("OmniStories")
        cls._logger.setLevel(logging.DEBUG)
        
        # File Handler (Detailed)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_fmt)
        
        cls._logger.addHandler(file_handler)

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
        """Print color-coded message and log to file."""
        # Ensure logging is initialized
        if not TerminalUI._logger:
            TerminalUI.setup_logging()
            
        c = color if color else ""
        print(f"{c}{text}{TerminalUI.NC}")
        
        # Strip ANSI codes for log file using regex
        clean_text = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', text)
        
        if color == TerminalUI.RED:
            TerminalUI._logger.error(clean_text)
        elif color == TerminalUI.YELLOW:
            TerminalUI._logger.warning(clean_text)
        else:
            TerminalUI._logger.info(clean_text)

    @staticmethod
    def input_prompt(text: str) -> str:
        """Display input prompt and log interaction."""
        if not TerminalUI._logger:
            TerminalUI.setup_logging()
            
        response = input(f"{TerminalUI.YELLOW}{text} {TerminalUI.NC}")
        TerminalUI._logger.info(f"PROMPT: {text} | RESPONSE: {response}")
        return response


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
    "thumbnail_font_size": 150,
    "thumbnail_font_color": "#FFFFFF",
    "thumbnail_font_shadow": False,
    "thumbnail_shadow_color": "#FFFFFF",
    "thumbnail_strength": "75%",
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
    "caption_strength": "75%",
    "caption_glow": True,
    "caption_glow_color": "#FFFFFF",
    "caption_glow_strength": "75%",
    "caption_glow_darken_factor": 0.75,

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


# Config cache to prevent redundant file reads
_config_cache: Optional[Dict[str, Any]] = None
_config_mtime: float = 0.0


def get_config(force_reload: bool = False) -> Dict[str, Any]:
    """
    Load and resolve application configuration with caching.
    
    Args:
        force_reload: If True, bypass cache and reload from disk.
    
    Returns:
        A dictionary containing the merged result of defaults and custom config.
    
    Raises:
        ConfigError: If config file exists but cannot be parsed.
    """
    global _config_cache, _config_mtime
    
    if yaml is None:
        return APP_DEFAULTS.copy()
    
    # Check if we can use cached config
    if not force_reload and _config_cache is not None:
        if CONFIG_PATH.exists():
            current_mtime = CONFIG_PATH.stat().st_mtime
            if current_mtime == _config_mtime:
                return _config_cache.copy()
        else:
            # Config file doesn't exist, cached defaults are still valid
            return _config_cache.copy()
    
    # Load config from disk
    config = APP_DEFAULTS.copy()
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, 'r') as f:
                loaded = yaml.safe_load(f)
                if loaded and isinstance(loaded, dict):
                    deep_merge(config, loaded)
            _config_mtime = CONFIG_PATH.stat().st_mtime
        except yaml.YAMLError as e:
            raise ConfigError(f"Failed to parse config.yaml: {e}")
        except Exception as e:
            raise ConfigError(f"Failed to load config.yaml: {e}")
    
    # Validate config values
    _validate_config(config)
    
    # Cache the config
    _config_cache = config.copy()
    return config


def _validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration values are within acceptable ranges.
    
    Args:
        config: The configuration dictionary to validate.
        
    Raises:
        ConfigError: If any config value is invalid.
    """
    # Validate quality
    valid_qualities = ["1080p", "2048p", "4096p"]
    if config.get("quality") not in valid_qualities:
        raise ConfigError(f"Invalid quality: {config.get('quality')}. Must be one of {valid_qualities}")
    
    # Validate aspect ratio
    valid_aspects = ["16:9", "9:16"]
    if config.get("aspect_ratio") not in valid_aspects:
        raise ConfigError(f"Invalid aspect_ratio: {config.get('aspect_ratio')}. Must be one of {valid_aspects}")
    
    # Validate background_fit
    valid_fits = ["crop", "stretch", "fit", "pad"]
    if config.get("background_fit") not in valid_fits:
        raise ConfigError(f"Invalid background_fit: {config.get('background_fit')}. Must be one of {valid_fits}")
    
    # Validate percentage values
    for key in ["background_volume", "narration_volume", "thumbnail_bg_blur", 
                "thumbnail_strength", "thumbnail_glow_strength", "caption_strength", "caption_glow_strength"]:
        value = str(config.get(key, "0%"))
        if isinstance(value, str) and value.endswith("%"):
            try:
                percent = float(value.replace("%", ""))
                if not 0 <= percent <= 100:
                    raise ConfigError(f"Invalid {key}: {value}. Percentage must be between 0% and 100%")
            except ValueError:
                raise ConfigError(f"Invalid {key}: {value}. Must be a valid percentage")
    
    # Validate font sizes
    for key in ["thumbnail_font_size", "caption_font_size"]:
        size = config.get(key, 0)
        if not isinstance(size, int) or size < 10 or size > 500:
            raise ConfigError(f"Invalid {key}: {size}. Must be an integer between 10 and 500")
    
    # Validate max_db_entries
    max_entries = config.get("max_db_entries", 500)
    if not isinstance(max_entries, int) or max_entries < 10:
        raise ConfigError(f"Invalid max_db_entries: {max_entries}. Must be an integer >= 10")



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
    
    _instance: Optional['KokoroEngine'] = None
    _lock = None  # Will be initialized on first use
    
    def __new__(cls):
        """Implement singleton pattern with thread-safe initialization."""
        if cls._instance is None:
            if cls._lock is None:
                import threading
                cls._lock = threading.Lock()
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self) -> None:
        """Initialize the ONNX inference session and load vocabulary."""
        if self._initialized:
            return
            
        TerminalUI.notify("Initializing Kokoro-TTS Engine...", TerminalUI.CYAN)
        try:
            import onnxruntime as ort
            import phonemizer
            
            model_path = Paths['kokoro'] / "model.onnx"
            vocab_path = Paths['kokoro'] / "tokenizer.json"
            
            if not model_path.exists():
                raise ModelError(f"Kokoro model not found at {model_path}")
            if not vocab_path.exists():
                raise ModelError(f"Kokoro tokenizer not found at {vocab_path}")
            
            self.session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
            with open(vocab_path, 'r') as f:
                self.vocab = json.load(f)['model']['vocab']
            self.voices = {}
            self.phonemizer = phonemizer
            TerminalUI.notify(f"{TerminalUI.SYM_OK} Kokoro-TTS Initialization Successful.")
            self.is_available = True
            self._initialized = True
        except ImportError as e:
            TerminalUI.notify(f"Kokoro-TTS Skipped: Missing dependency ({e})", TerminalUI.YELLOW)
            self.is_available = False
            self._initialized = True
        except (ModelError, FileNotFoundError) as e:
            TerminalUI.notify(f"Kokoro-TTS Model Error: {e}", TerminalUI.RED)
            self.is_available = False
            self._initialized = True
        except Exception as e:
            TerminalUI.notify(f"Kokoro-TTS Initialization Failed: {e}", TerminalUI.RED)
            self.is_available = False
            self._initialized = True

    def get_voice_embedding(self, name: str) -> Optional[Any]:
        """
        Load voice characteristics from binary embedding files.
        
        Args:
            name: Identifier for the requested voice.
            
        Returns:
            NumPy array containing the voice embedding data.
            
        Raises:
            ModelError: If voice file cannot be loaded.
        """
        import numpy as np
        if name not in self.voices:
            path = Paths['kokoro'] / f"{name}.bin"
            if not path.exists():
                path = Paths['kokoro'] / "am_adam.bin"
            if not path.exists():
                raise ModelError(f"Voice embedding not found: {name}")
            try:
                raw_data = np.fromfile(path, dtype=np.float32)
                self.voices[name] = raw_data.reshape(-1, 1, 256)
            except Exception as e:
                raise ModelError(f"Failed to load voice embedding {name}: {e}")
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
            
        Raises:
            ModelError: If synthesis fails.
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
        try:
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
        except Exception as e:
            raise ModelError(f"TTS synthesis failed: {e}")


def generate_local_audio(text: str, output_path: str) -> str:
    """
    Generate audio using local Kokoro TTS engine.
    Uses singleton pattern for efficient resource management.
    
    Args:
        text: Text to synthesize.
        output_path: Output file path.
        
    Returns:
        Path to generated audio file.
        
    Raises:
        ModelError: If TTS generation fails.
        DependencyError: If Kokoro engine is unavailable.
    """
    engine = KokoroEngine()
    
    if not engine.is_available:
        raise DependencyError("Kokoro TTS unavailable. Cannot generate audio without API keys or local TTS.")

    config = get_config()
    try:
        return engine.synthesize(text, config.get("kokoro_voice", "am_adam"), output_path)
    except ModelError:
        raise
    except Exception as e:
        raise ModelError(f"Kokoro Generation Failed: {e}")





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

    def tokenize(text: str) -> List[str]:
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
        
    Raises:
        APIError: If all API keys fail.
        DependencyError: If no TTS method is available.
    """
    api_key_pool = [k.strip() for k in os.getenv("ELEVENLABS_API_KEYS", "").split(",") if k.strip() and k.strip().lower() != "none"]
    config = get_config()
    voice_id = config.get("voice_id", "pNInz6obpgDQGcFmaJgB")
    
    model_id = "eleven_multilingual_v2"
    if config.get("voice_model") == "v3":
        model_id = "eleven_multilingual_v3"
        
    final_output = str(Paths['tts_generations'] / f"{destination_name}.mp3")

    last_error = None
    for index, clean_key in enumerate(api_key_pool):
        TerminalUI.notify(f"Attempting ElevenLabs (Key {index+1}/{len(api_key_pool)})...", TerminalUI.CYAN)
        
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
                    TerminalUI.notify(f"{TerminalUI.SYM_OK} ElevenLabs generation successful", TerminalUI.GREEN)
                    return final_output
                else:
                    # Read full error message
                    f.seek(0)
                    error_payload = f.read(1024).decode('utf-8', errors='ignore')
                    last_error = error_payload
                    TerminalUI.notify(f"ElevenLabs Error (Key {index+1}): {error_payload}", TerminalUI.YELLOW)
            
            if os.path.exists(final_output):
                os.remove(final_output)
        
        # Exponential backoff between attempts (only if not last key)
        if index < len(api_key_pool) - 1:
            delay = min(2 ** index, 8)  # Cap at 8 seconds
            TerminalUI.notify(f"Waiting {delay}s before next attempt...", TerminalUI.CYAN)
            time.sleep(delay)

    # All API keys failed, fallback to local TTS
    if api_key_pool:
        TerminalUI.notify(f"All ElevenLabs keys exhausted. Falling back to local TTS...", TerminalUI.YELLOW)
    else:
        TerminalUI.notify("No ElevenLabs API keys configured. Using local TTS...", TerminalUI.CYAN)
    
    local_output = str(Paths['tts_generations'] / f"{destination_name}.wav")
    try:
        return generate_local_audio(text_content, local_output)
    except DependencyError:
        raise
    except Exception as e:
        raise ModelError(f"Local TTS generation failed: {e}")


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

def get_system_font_paths() -> List[str]:
    """
    Get common system font directories based on OS.
    
    Returns:
        List of directory paths where fonts might be found.
    """
    paths = []
    if sys.platform == "darwin":
        paths = ["/System/Library/Fonts", "/Library/Fonts", str(Path.home() / "Library/Fonts")]
    elif sys.platform == "win32":
        windir = os.environ.get("WINDIR", "C:\\Windows")
        paths = [os.path.join(windir, "Fonts"), os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft\\Windows\\Fonts")]
    elif sys.platform.startswith("linux"):
        paths = [
            "/usr/share/fonts", "/usr/local/share/fonts", 
            str(Path.home() / ".local/share/fonts"), str(Path.home() / ".fonts")
        ]
    return [p for p in paths if os.path.isdir(p)]


def resolve_system_font(name: str, size: int) -> ImageFont.FreeTypeFont:
    """
    Locate a system font by name with cross-platform fallback logic.
    
    Args:
        name: The requested font family name.
        size: The desired font point size.
        
    Returns:
        A loaded ImageFont object.
    """
    # 1. Try direct loading (works if font is installed and PIL can find it)
    try:
        return ImageFont.truetype(name, size)
    except Exception:
        pass
        
    # 2. Try common extensions
    for ext in [".ttf", ".otf", ".ttc"]:
        try:
            return ImageFont.truetype(f"{name}{ext}", size)
        except Exception:
            pass

    # 3. Platform-specific resolution
    if sys.platform.startswith("linux") and shutil.which("fc-match"):
        try:
            result = subprocess.run(["fc-match", "-f", "%{file}", name], capture_output=True, text=True, timeout=2)
            if result.returncode == 0 and result.stdout.strip():
                font_file = result.stdout.strip()
                if os.path.exists(font_file):
                    return ImageFont.truetype(font_file, size)
        except Exception:
            pass
            
    # 4. Brute force search in system directories (fallback for Windows/Mac if simple load failed)
    font_files = []
    lower_name = name.lower()
    for directory in get_system_font_paths():
        try:
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith(('.ttf', '.otf', '.ttc')):
                        if lower_name in file.lower():
                            font_files.append(os.path.join(root, file))
        except Exception:
            continue
            
    # Try found candidates
    for font_path in font_files:
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            pass
            
    # 5. Last resort fallback
    try:
        # Try a known universal default if possible, otherwise PIL default
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def resolve_emoji_font(size: int) -> Optional[ImageFont.FreeTypeFont]:
    """Locate a color emoji font on the system."""
    # Priority list based on OS
    if sys.platform == "darwin":
        emoji_fonts = ["Apple Color Emoji", "AppleColorEmoji.ttc"]
    elif sys.platform == "win32":
        emoji_fonts = ["Segoe UI Emoji", "seguiemj.ttf"]
    else:
        emoji_fonts = ["Noto Color Emoji", "NotoColorEmoji.ttf", "Apple Color Emoji", "Segoe UI Emoji"]
        
    for name in emoji_fonts:
        # Try requested size first, then common bitmap sizes (109) and powers of 2
        for target_size in [size, 109, 128]:
            try:
                return ImageFont.truetype(name, target_size)
            except Exception:
                pass
        
        # Check system paths specifically for these files
        for font_dir in get_system_font_paths():
            possible_path = os.path.join(font_dir, name)
            if os.path.exists(possible_path):
                 for target_size in [size, 109, 128]:
                    try:
                        return ImageFont.truetype(possible_path, target_size)
                    except Exception:
                        pass
        
        # Linux fontconfig fallback
        if sys.platform.startswith("linux") and shutil.which("fc-match"):
            try:
                result = subprocess.run(["fc-match", "-f", "%{file}", name], capture_output=True, text=True, timeout=2)
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
        str(Paths['data'] / "bin" / "ffmpeg"),
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
            
    # Priority: First binary with libass support
    for entry in unique_paths:
        try:
            probe = subprocess.run([entry, "-version"], capture_output=True, text=True, check=False)
            if "--enable-libass" in probe.stdout:
                return entry
        except Exception:
            continue
            
    # Fallback to first available if none have libass (though this will likely fail later)
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


def sanitize_slug(text: str) -> str:
    """
    Sanitize text to create a safe filesystem slug.
    
    Args:
        text: Input text to sanitize.
        
    Returns:
        Safe filesystem slug.
        
    Raises:
        ValidationError: If resulting slug is invalid.
    """
    # Remove/replace dangerous characters
    slug = "".join(c for c in text if c.isalnum() or c in " -_").strip()
    slug = slug.replace(" ", "_")
    
    # Prevent directory traversal
    if ".." in slug or "/" in slug or "\\" in slug:
        raise ValidationError("Invalid characters in title")
    
    # Ensure not empty
    if not slug:
        raise ValidationError("Title cannot be empty after sanitization")
    
    # Limit length
    if len(slug) > 200:
        slug = slug[:200]
    
    return slug


def _prepare_video_assets(title: str) -> Tuple[str, Path, Path]:
    """
    Validate inputs and prepare workspace.
    
    Returns:
        Tuple of (safe_slug, destination_dir, chosen_bg_path).
    """
    try:
        safe_slug = sanitize_slug(title)
        destination_dir = Paths['output'] / safe_slug
        
        if destination_dir.exists():
            raise FileSystemError(f"Directory already exists: {destination_dir}")
            
        available_bgs = list(Paths['background_videos'].glob("*.mp4"))
        if not available_bgs:
            raise DependencyError(f"No background videos found in {Paths['background_videos']}")
            
        chosen_bg = random.choice(available_bgs)
        destination_dir.mkdir(parents=True, exist_ok=True)
        
        return safe_slug, destination_dir, chosen_bg
        
    except (ValidationError, FileSystemError, DependencyError):
        raise
    except Exception as e:
        raise FileSystemError(f"Failed to prepare workspace: {e}")


def _generate_story_assets(content: str, safe_slug: str, aspect_ratio: str) -> Tuple[str, Path]:
    """
    Generate audio and subtitles for the story.
    
    Returns:
        Tuple of (audio_path, ass_path).
    """
    audio_path = generate_vocal_audio(content, safe_slug)
    aligned_words = align_captions(audio_path, content)
    
    ass_path = Paths['tts_generations'] / f"{safe_slug}.ass"
    generate_ass_subtitles(aligned_words, str(ass_path), aspect_ratio)
    
    return audio_path, ass_path


def _finalize_video_production(
    title: str, url: str, content: str, quote: str, tags: str,
    destination_dir: Path, safe_slug: str, aspect_ratio: str,
    chosen_bg: Path, audio_path: str, ass_path: Path
) -> None:
    """Compose video, create thumbnail, and record metadata."""
    
    # Calculate offset logic
    video_meta = get_probe_data(str(chosen_bg))
    video_dur = float(video_meta.get('format', {}).get('duration', 0))
    audio_meta = get_probe_data(audio_path)
    audio_dur = float(audio_meta.get('format', {}).get('duration', 0))

    start_offset = 0.0
    if video_dur > audio_dur + 2:
        start_offset = random.uniform(0, video_dur - audio_dur - 2)

    video_path = str(destination_dir / f"{safe_slug}.mp4")
    compose_final_video(str(chosen_bg), audio_path, str(ass_path), video_path, aspect_ratio, start_offset)
    
    # Generate thumbnail
    ThumbnailGenerator.create(str(chosen_bg), quote or title, str(destination_dir / "thumbnail.png"), start_offset + 1.0)
    
    # Write metadata
    with open(destination_dir / "metadata.md", 'w') as f:
        f.write(f"Title: {title}\n")
        f.write(f"URL: {url}\n")
        f.write(f"Date: {time.ctime()}\n\n")
        f.write(f"{content}")

    # Verify integrity
    success, message = verify_output_integrity(video_path)
    if not success:
        raise FileSystemError(f"Verification failed: {message}")

    # Record to DB
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


def execute_generation_pipeline(url: str, title: str, content: str, aspect_ratio: str, quote: str = "", tags: str = "") -> None:
    """
    Execute the full end-to-end video production workflow.
    Refactored to use modular phases for better maintainability.
    """
    destination_dir = None
    
    try:
        # Phase 1: Preparation
        safe_slug, destination_dir, chosen_bg = _prepare_video_assets(title)
        
        TerminalUI.notify(f"Starting production: {TerminalUI.BOLD}{title}{TerminalUI.NC}", TerminalUI.CYAN)
        
        # Phase 2: Asset Generation
        audio_path, ass_path = _generate_story_assets(content, safe_slug, aspect_ratio)
        
        # Phase 3: Finalization
        _finalize_video_production(
            title, url, content, quote, tags,
            destination_dir, safe_slug, aspect_ratio,
            chosen_bg, audio_path, ass_path
        )
        
        TerminalUI.notify(f"{TerminalUI.SYM_OK} Generation complete: {destination_dir}", TerminalUI.GREEN)
        
    except (ValidationError, FileSystemError, DependencyError, APIError, ModelError) as e:
        TerminalUI.notify(f"{TerminalUI.SYM_ERR} Generation failed: {e}", TerminalUI.RED)
        if destination_dir and destination_dir.exists():
            import shutil
            shutil.rmtree(destination_dir)
            TerminalUI.notify("Cleaned up partial generation", TerminalUI.YELLOW)
    except Exception as e:
        TerminalUI.notify(f"{TerminalUI.SYM_ERR} Unexpected error: {e}", TerminalUI.RED)
        import traceback
        TerminalUI.notify(traceback.format_exc(), TerminalUI.RED)
        if destination_dir and destination_dir.exists():
            import shutil
            shutil.rmtree(destination_dir)


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


def _setup_credentials(api_key: Optional[str]) -> None:
    """Configure API credentials in .env file."""
    if api_key and api_key != "NO_KEY":
        set_key(ENV_PATH, "ELEVENLABS_API_KEYS", api_key.strip())
        load_dotenv(ENV_PATH, override=True)
        TerminalUI.notify(f"{TerminalUI.SYM_OK} Credentials configured.")

def _install_pip_dependencies() -> None:
    """Check and install missing Python packages."""
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

def _create_global_symlink() -> None:
    """Create a global executable command in user's bin directory."""
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

def _init_project_folders() -> None:
    """Create necessary project subdirectories."""
    for folder in [Paths['data'], Paths['output'], Paths['background_videos'], Paths['models'], Paths['tts_generations']]:
        folder.mkdir(parents=True, exist_ok=True)

def run_setup_routine(api_key: Optional[str] = None) -> None:
    """
    Configure the application environment and install dependencies.
    
    Args:
        api_key: Optional ElevenLabs API key for initial configuration.
    """
    TerminalUI.print_large_banner()
    TerminalUI.notify("Initializing Omni-Stories Environment...", TerminalUI.CYAN)
    
    _setup_credentials(api_key)
    _install_pip_dependencies()
    _create_global_symlink()
    _init_project_folders()
        
    TerminalUI.notify(f"\n{TerminalUI.GREEN}Environment Setup Complete.{TerminalUI.NC}")


def check_binary_version(binary: str) -> Optional[str]:
    """
    Get version information for a system binary.
    
    Args:
        binary: Binary name to check.
        
    Returns:
        Version string or None if unavailable.
    """
    try:
        result = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=5)
        # Extract first line of version output
        return result.stdout.split('\n')[0] if result.stdout else None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


def check_ffmpeg_capabilities() -> Dict[str, bool]:
    """
    Test FFmpeg for required capabilities.
    
    Returns:
        Dictionary of capability names and their availability.
    """
    capabilities = {
        "libass": False,  # Subtitle rendering
        "h264": False,    # Video codec
        "aac": False,     # Audio codec
    }
    
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5)
        output = result.stdout.lower()
        
        capabilities["libass"] = "--enable-libass" in output
        
        # Check codecs
        result = subprocess.run(["ffmpeg", "-codecs"], capture_output=True, text=True, timeout=5)
        codec_output = result.stdout.lower()
        capabilities["h264"] = "h264" in codec_output
        capabilities["aac"] = "aac" in codec_output
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    
    return capabilities


def verify_model_integrity(model_path: Path, min_size: int = 100000) -> bool:
    """
    Verify model file integrity.
    
    Args:
        model_path: Path to model file.
        min_size: Minimum expected file size in bytes.
        
    Returns:
        True if model appears valid.
    """
    if not model_path.exists():
        return False
    
    try:
        size = model_path.stat().st_size
        return size >= min_size
    except OSError:
        return False


def check_disk_space(path: Path, required_mb: int = 500) -> Tuple[bool, int]:
    """
    Check available disk space.
    
    Args:
        path: Path to check.
        required_mb: Required space in MB.
        
    Returns:
        Tuple of (has_enough_space, available_mb).
    """
    try:
        import shutil
        stat = shutil.disk_usage(path)
        available_mb = stat.free // (1024 * 1024)
        return (available_mb >= required_mb, available_mb)
    except OSError:
        return (False, 0)


def _diagnose_python_env(issues: List[str], healable: List[str]) -> None:
    """Diagnose Python version and dependencies."""
    TerminalUI.notify("[1/9] Checking Python Environment...", TerminalUI.CYAN)
    
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    TerminalUI.notify(f"  Python Version: {py_version}", TerminalUI.WHITE)
    
    if sys.version_info < (3, 9):
        issues.append("Python 3.9+ required")
        TerminalUI.notify(f"  {TerminalUI.SYM_ERR} Python 3.9+ required", TerminalUI.RED)
    elif sys.version_info >= (3, 13):
        TerminalUI.notify(f"  {TerminalUI.SYM_WARN} Python {py_version} is experimental. Some features (Kokoro TTS) may fail.", TerminalUI.YELLOW)
    else:
        TerminalUI.notify(f"  {TerminalUI.SYM_OK} Python version compatible", TerminalUI.GREEN)
    
    missing_deps = verify_python_environment()
    if missing_deps:
        issues.append(f"{len(missing_deps)} missing Python packages")
        healable.append("missing_deps")
        TerminalUI.notify(f"  {TerminalUI.SYM_WARN} Missing packages: {', '.join(missing_deps)}", TerminalUI.YELLOW)
    else:
        TerminalUI.notify(f"  {TerminalUI.SYM_OK} All Python dependencies installed", TerminalUI.GREEN)
    print()


def _diagnose_binaries(issues: List[str], healable: List[str]) -> None:
    """Check system binaries availability and capabilities."""
    TerminalUI.notify("[2/9] Checking System Binaries...", TerminalUI.CYAN)
    
    binaries = {
        "ffmpeg": "Video composition",
        "ffprobe": "Metadata extraction",
        "espeak-ng": "Phoneme generation",
        "curl": "File downloads"
    }
    
    for binary, description in binaries.items():
        if not shutil.which(binary):
            issues.append(f"Missing binary: {binary}")
            TerminalUI.notify(f"  {TerminalUI.SYM_ERR} {binary} not found ({description})", TerminalUI.RED)
        else:
            version = check_binary_version(binary)
            version_str = f" - {version}" if version else ""
            TerminalUI.notify(f"  {TerminalUI.SYM_OK} {binary}{version_str}", TerminalUI.GREEN)
    
    if shutil.which("ffmpeg"):
        caps = check_ffmpeg_capabilities()
        if not caps["libass"]:
            issues.append("FFmpeg missing libass support")
            healable.append("ffmpeg_libass")
            TerminalUI.notify(f"  {TerminalUI.SYM_WARN} FFmpeg: libass (subtitles) not enabled", TerminalUI.YELLOW)
        else:
            TerminalUI.notify(f"  {TerminalUI.SYM_OK} FFmpeg: libass enabled", TerminalUI.GREEN)
        
        if not caps["h264"]:
            issues.append("FFmpeg missing H.264 codec")
        if not caps["aac"]:
            issues.append("FFmpeg missing AAC codec")
    print()


def _diagnose_storage(issues: List[str], healable: List[str]) -> None:
    """Check disk space and directory structure."""
    TerminalUI.notify("[3/9] Checking Disk Space...", TerminalUI.CYAN)
    
    has_space, available_mb = check_disk_space(Paths['data'], required_mb=500)
    if not has_space:
        issues.append(f"Low disk space ({available_mb}MB)")
        TerminalUI.notify(f"  {TerminalUI.SYM_WARN} Low disk space: {available_mb}MB", TerminalUI.YELLOW)
    else:
        TerminalUI.notify(f"  {TerminalUI.SYM_OK} Disk space: {available_mb}MB available", TerminalUI.GREEN)
    print()
    
    TerminalUI.notify("[4/9] Checking Directory Structure...", TerminalUI.CYAN)
    missing_dirs = []
    for key, path in Paths.items():
        if key != 'database' and not path.exists():
            missing_dirs.append(key)
            healable.append(f"missing_dir:{key}")
    
    if missing_dirs:
        issues.append(f"{len(missing_dirs)} missing directories")
        for key in missing_dirs:
            path = Paths[key]
            path.mkdir(parents=True, exist_ok=True)
            TerminalUI.notify(f"  {TerminalUI.SYM_OK} Healed: {path}", TerminalUI.YELLOW)
    else:
        TerminalUI.notify(f"  {TerminalUI.SYM_OK} All directories exist", TerminalUI.GREEN)
    print()


def _diagnose_models(issues: List[str], healable: List[str]) -> None:
    """Verify ML models integrity."""
    TerminalUI.notify("[5/9] Checking Model Files...", TerminalUI.CYAN)
    
    kokoro_model = Paths['kokoro'] / "model.onnx"
    kokoro_vocab = Paths['kokoro'] / "tokenizer.json"
    kokoro_voice = Paths['kokoro'] / "am_adam.bin"
    
    if not verify_model_integrity(kokoro_model, min_size=100000000):
        issues.append("Kokoro model missing or corrupted")
        healable.append("kokoro_model")
        TerminalUI.notify(f"  {TerminalUI.SYM_WARN} Kokoro model missing/corrupted", TerminalUI.YELLOW)
    else:
        size_mb = kokoro_model.stat().st_size // (1024 * 1024)
        TerminalUI.notify(f"  {TerminalUI.SYM_OK} Kokoro model ({size_mb}MB)", TerminalUI.GREEN)
    
    if not kokoro_vocab.exists():
        issues.append("Kokoro tokenizer missing")
        healable.append("kokoro_vocab")
        TerminalUI.notify(f"  {TerminalUI.SYM_WARN} Kokoro tokenizer missing", TerminalUI.YELLOW)
    else:
        TerminalUI.notify(f"  {TerminalUI.SYM_OK} Kokoro tokenizer", TerminalUI.GREEN)
    
    if not kokoro_voice.exists():
        issues.append("Kokoro voice embedding missing")
        healable.append("kokoro_voice")
        TerminalUI.notify(f"  {TerminalUI.SYM_WARN} Kokoro voice embedding missing", TerminalUI.YELLOW)
    else:
        TerminalUI.notify(f"  {TerminalUI.SYM_OK} Kokoro voice embedding", TerminalUI.GREEN)
    print()


def _diagnose_config_and_api(issues: List[str]) -> None:
    """Validate configuration and API keys."""
    TerminalUI.notify("[6/9] Checking Configuration...", TerminalUI.CYAN)
    try:
        config = get_config(force_reload=True)
        TerminalUI.notify(f"  {TerminalUI.SYM_OK} Config file valid", TerminalUI.GREEN)
        TerminalUI.notify(f"    Quality: {config.get('quality')}", TerminalUI.WHITE)
        TerminalUI.notify(f"    Aspect: {config.get('aspect_ratio')}", TerminalUI.WHITE)
    except ConfigError as e:
        issues.append(f"Config error: {e}")
        TerminalUI.notify(f"  {TerminalUI.SYM_ERR} Config invalid: {e}", TerminalUI.RED)
    print()
    
    TerminalUI.notify("[7/9] Checking API Credentials...", TerminalUI.CYAN)
    keys = os.getenv("ELEVENLABS_API_KEYS", "")
    if not keys or "placeholder" in keys.lower() or keys.strip().lower() == "none":
        TerminalUI.notify(f"  {TerminalUI.SYM_WARN} No ElevenLabs API keys (using local TTS only)", TerminalUI.YELLOW)
    else:
        key_count = len([k for k in keys.split(',') if k.strip()])
        TerminalUI.notify(f"  {TerminalUI.SYM_OK} {key_count} ElevenLabs API key(s) configured", TerminalUI.GREEN)
    print()


def _diagnose_data_assets(issues: List[str], healable: List[str]) -> None:
    """Check database and background videos."""
    TerminalUI.notify("[8/9] Checking Database...", TerminalUI.CYAN)
    try:
        db = load_database()
        story_count = len(db.get('stories', []))
        TerminalUI.notify(f"  {TerminalUI.SYM_OK} Database valid ({story_count} entries)", TerminalUI.GREEN)
    except Exception as e:
        issues.append("Database corrupted")
        healable.append("database")
        TerminalUI.notify(f"  {TerminalUI.SYM_ERR} Database corrupted: {e}", TerminalUI.RED)
    print()
    
    TerminalUI.notify("[9/9] Checking Background Videos...", TerminalUI.CYAN)
    bgs = list(Paths['background_videos'].glob("*.mp4"))
    if not bgs:
        issues.append("No background videos")
        TerminalUI.notify(f"  {TerminalUI.SYM_WARN} No background videos found", TerminalUI.YELLOW)
        TerminalUI.notify(f"    Tip: omni-stories --dl_video -u <youtube_url> -r 1080p", TerminalUI.WHITE)
    else:
        total_size = sum(bg.stat().st_size for bg in bgs) // (1024 * 1024)
        TerminalUI.notify(f"  {TerminalUI.SYM_OK} {len(bgs)} background video(s) ({total_size}MB)", TerminalUI.GREEN)
    print()


def _perform_auto_healing(healable_issues: List[str]) -> None:
    """Execute healing routines for identified issues."""
    TerminalUI.notify("\nInitiating auto-heal...", TerminalUI.CYAN)
    
    if "missing_deps" in healable_issues:
        TerminalUI.notify("Installing missing Python packages...", TerminalUI.CYAN)
        try:
            missing = verify_python_environment()
            pip_cmd = [sys.executable, "-m", "pip", "install"]
            
            # Handle PEP 668 managed environments
            if sys.prefix == sys.base_prefix:
                pip_cmd.append("--user")
                if sys.version_info >= (3, 11):
                    # Python 3.11+ often requires this on Linux/macOS
                    pip_cmd.append("--break-system-packages")
            
            pip_cmd.extend(["-q"] + missing)
            
            subprocess.check_call(pip_cmd)
            TerminalUI.notify(f"{TerminalUI.SYM_OK} Dependencies installed", TerminalUI.GREEN)
        except subprocess.CalledProcessError as e:
            # Check if critical
            TerminalUI.notify(f"{TerminalUI.SYM_WARN} Partial install failure: {e}", TerminalUI.YELLOW)
            TerminalUI.notify("Hint: 'onnxruntime' failure on Python 3.13+ is expected and non-critical (local TTS will be disabled).", TerminalUI.WHITE)
    
    if "kokoro_model" in healable_issues:
        TerminalUI.notify("Downloading Kokoro model...", TerminalUI.CYAN)
        url = "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v0_19.onnx"
        try:
            target = Paths['kokoro'] / "model.onnx"
            subprocess.run(["curl", "-L", "-#", url, "--output", str(target)], check=True, timeout=300)
            TerminalUI.notify(f"{TerminalUI.SYM_OK} Model downloaded", TerminalUI.GREEN)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            TerminalUI.notify(f"{TerminalUI.SYM_ERR} Download failed: {e}", TerminalUI.RED)
            
    if "database" in healable_issues:
        TerminalUI.notify("Repairing database...", TerminalUI.CYAN)
        db_path = Paths['database']
        if db_path.exists():
            backup_path = db_path.parent / f"database.backup.{int(time.time())}.json"
            import shutil as sh
            sh.copy(db_path, backup_path)
            TerminalUI.notify(f"  Backup created: {backup_path}", TerminalUI.WHITE)
        
        save_database({"stories": []})
        TerminalUI.notify(f"{TerminalUI.SYM_OK} Database reset", TerminalUI.GREEN)
    
    if "ffmpeg_libass" in healable_issues:
        TerminalUI.notify("Downloading static FFmpeg (with libass)...", TerminalUI.CYAN)
        bin_dir = Paths['data'] / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        
        # Linux amd64 static build (most likely environment)
        # For a truly universal script, we'd need arch detection here, but user is on Linux Mint.
        url = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
        tar_path = bin_dir / "ffmpeg.tar.xz"
        
        try:
            TerminalUI.notify(f"  Fetching from {url}...", TerminalUI.WHITE)
            subprocess.run(["curl", "-L", "-#", url, "--output", str(tar_path)], check=True, timeout=600)
            
            TerminalUI.notify("  Extracting...", TerminalUI.WHITE)
            subprocess.run(["tar", "-xf", str(tar_path), "--strip-components=1", "-C", str(bin_dir), "--wildcards", "*/ffmpeg", "*/ffprobe"], check=True)
            
            # Cleanup
            if tar_path.exists():
                tar_path.unlink()
                
            # Verify
            if (bin_dir / "ffmpeg").exists():
                (bin_dir / "ffmpeg").chmod(0o755)
                TerminalUI.notify(f"{TerminalUI.SYM_OK} Static FFmpeg installed successfully", TerminalUI.GREEN)
            else:
                TerminalUI.notify(f"{TerminalUI.SYM_ERR} Extraction failed", TerminalUI.RED)
                
        except Exception as e:
             TerminalUI.notify(f"{TerminalUI.SYM_ERR} Static FFmpeg download failed: {e}", TerminalUI.RED)
             TerminalUI.notify("  Please install generic ffmpeg manually: sudo apt install ffmpeg", TerminalUI.YELLOW)

    TerminalUI.notify(f"\n{TerminalUI.SYM_OK} Auto-heal complete. Re-run --doctor to verify.", TerminalUI.GREEN)


def run_health_check() -> None:
    """
    Perform comprehensive system diagnostic with automated self-healing.
    Modularized for better maintenance.
    """
    TerminalUI.print_large_banner()
    TerminalUI.notify("═══ SYSTEM DIAGNOSTIC ═══", TerminalUI.CYAN)
    print()
    
    issues_found = []
    healable_issues = []
    
    # Run diagnostic modules
    _diagnose_python_env(issues_found, healable_issues)
    _diagnose_binaries(issues_found, healable_issues)
    _diagnose_storage(issues_found, healable_issues)
    _diagnose_models(issues_found, healable_issues)
    _diagnose_config_and_api(issues_found)
    _diagnose_data_assets(issues_found, healable_issues)
    
    # Summary & Healing
    print("═" * 60)
    if not issues_found:
        TerminalUI.notify(f"{TerminalUI.SYM_OK} ALL SYSTEMS NOMINAL", TerminalUI.GREEN)
        TerminalUI.notify("System is fully operational.", TerminalUI.WHITE)
    else:
        TerminalUI.notify(f"{TerminalUI.SYM_WARN} {len(issues_found)} ISSUE(S) DETECTED", TerminalUI.YELLOW)
        for issue in issues_found:
            TerminalUI.notify(f"  • {issue}", TerminalUI.WHITE)
        
        if healable_issues:
            # Automated healing - No prompt requested by user
            _perform_auto_healing(healable_issues)
    
    print("═" * 60)
    print()


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
