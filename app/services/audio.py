_whisper_model = None

from app.core.logger import get_logger
logger = get_logger(__name__)

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        logger.info("[Models] Loading Whisper 'medium' (CPU int8)...")
        _whisper_model = WhisperModel("medium", device="cpu", compute_type="int8")
    return _whisper_model

def transcribe_audio_file(temp_path: str) -> str:
    model = get_whisper_model()
    # Transcribe the audio (beam_size=1 is faster)
    segments, info = model.transcribe(temp_path, beam_size=1)
    transcription = "".join([segment.text for segment in segments]).strip()
    return transcription
