import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from litellm import completion

from stt import audio_to_text
from text_extract import IMAGE_EXTS, extract_text

load_dotenv()

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm", ".aac", ".wma"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv"}
DEFAULT_PROMPT = "You are a helpful assistant."


def _parse_chain(raw: str) -> list[str]:
    chain = [entry.strip() for entry in raw.split(",") if entry.strip()]
    if not chain:
        raise ValueError("LLM_CHAIN is empty in .env")
    for entry in chain:
        if "/" not in entry:
            raise ValueError(
                f"Invalid LLM_CHAIN entry '{entry}' (expected provider/model)"
            )
    return chain


def _parse_api_keys(raw: str) -> dict[str, str]:
    keys = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(
                f"Invalid LLM_API_KEYS entry '{part}' (expected provider=key)"
            )
        provider, key = part.split("=", 1)
        keys[provider.strip()] = key.strip()
    return keys


def _config() -> dict:
    chain = _parse_chain(os.getenv("LLM_CHAIN", ""))
    api_keys = _parse_api_keys(os.getenv("LLM_API_KEYS", ""))
    system_prompt = os.getenv(
        "LLM_SYSTEM_PROMPT", "You are a helpful assistant."
    ).strip()
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "2048"))

    for entry in chain:
        provider = entry.split("/", 1)[0]
        if provider not in api_keys or not api_keys[provider]:
            raise ValueError(f"Missing API key for provider '{provider}' in LLM_API_KEYS")

    return {
        "chain": chain,
        "api_keys": api_keys,
        "system_prompt": system_prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def get_system_prompt(file_path: str) -> str:
    """Pick the system prompt for this file type based on its extension."""
    ext = Path(file_path).suffix.lower()
    if ext in IMAGE_EXTS:
        var = "prompt_img"
    elif ext in VIDEO_EXTS:
        var = "prompt_video"
    elif ext in AUDIO_EXTS:
        var = "prompt_audio"
    else:
        var = "LLM_SYSTEM_PROMPT"

    default = os.getenv("LLM_SYSTEM_PROMPT", DEFAULT_PROMPT).strip()
    return (os.getenv(var) or default).strip() or default


def _video_to_text(file_path: str, whisper_model: str) -> str:
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", file_path,
                "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", wav_path,
            ],
            check=True,
            capture_output=True,
        )
        return audio_to_text(wav_path, whisper_model)
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass


def get_input_text(file_path: str) -> Optional[str]:
    """Transcribe audio/video or extract text from image/PDF/TXT/DOCX."""
    ext = Path(file_path).suffix.lower()
    model = os.getenv("WHISPER_MODEL", "base")
    if ext in AUDIO_EXTS:
        return audio_to_text(file_path, model)
    if ext in VIDEO_EXTS:
        return _video_to_text(file_path, model)
    return extract_text(file_path)


def ask_llm(user_text: str, system_prompt: Optional[str] = None) -> str:
    """Send text down the LLM_CHAIN, one attempt per model, until one succeeds."""
    cfg = _config()
    if system_prompt is None:
        system_prompt = cfg["system_prompt"]
    errors: list[str] = []

    for entry in cfg["chain"]:
        provider = entry.split("/", 1)[0]
        try:
            response = completion(
                model=entry,
                api_key=cfg["api_keys"][provider],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
                temperature=cfg["temperature"],
                max_tokens=cfg["max_tokens"],
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from model")
            return content
        except Exception as exc:
            errors.append(f"{entry}: {exc}")
            continue

    raise RuntimeError("All models in LLM_CHAIN failed:\n" + "\n".join(errors))


def run_pipeline(file_path: str) -> Optional[str]:
    """Full pipeline: file -> text -> LLM (prompt picked by file type)."""
    text = get_input_text(file_path)
    if not text:
        return None
    return ask_llm(text, get_system_prompt(file_path))
