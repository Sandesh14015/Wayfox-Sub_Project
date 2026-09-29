import whisper

_model_cache = {}


def load_model(model_name: str = "base"):
    if model_name not in _model_cache:
        _model_cache[model_name] = whisper.load_model(model_name)
    return _model_cache[model_name]


def audio_to_text(audio_path: str, model_name: str = "base") -> str:
    """Convert an audio file to text using OpenAI Whisper.

    Args:
        audio_path: Path to the audio file (mp3, wav, m4a, etc.).
        model_name: Whisper model size (tiny, base, small, medium, large).

    Returns:
        Transcribed text as a string.
    """
    model = load_model(model_name)
    result = model.transcribe(audio_path)
    return result["text"].strip()
