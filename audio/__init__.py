"""
audio_lib - High-Level Audio-Bibliothek für TTS und STT

Eine einfache Python-Bibliothek für Text-to-Speech und Speech-to-Text
mit OpenAI's Whisper und TTS-Modellen.

Features:
- Speech-to-Text (STT) mit Whisper
- Text-to-Speech (TTS) mit verschiedenen Stimmen
- Multi-Speaker Audio-Generierung aus Textdateien
- Audio-Analyse und Sentiment-Analyse
- Podcast-Generierung
- Audio-Übersetzung
- Audio-Kombinierung

Autor: Ralf
Version: 1.0
Jahr: 2025
"""

from .config import AudioConfig
from .audio_lib import (
    # STT Funktionen
    transcribe_audio,
    transcribe_with_timestamps,

    # TTS Funktionen
    text_to_speech,
    create_multi_speaker_audio,

    # Utilities
    combine_audio_files,

    # Audio-Analyse
    create_audio_summary,
    analyze_audio_sentiment,

    # Podcast
    create_podcast,
    create_podcast_with_random_texts,

    # Übersetzung
    translate_audio,
)

__version__ = "1.0.0"
__author__ = "Ralf"

__all__ = [
    # Config
    "AudioConfig",

    # STT
    "transcribe_audio",
    "transcribe_with_timestamps",

    # TTS
    "text_to_speech",
    "create_multi_speaker_audio",

    # Utilities
    "combine_audio_files",

    # Analyse
    "create_audio_summary",
    "analyze_audio_sentiment",

    # Podcast
    "create_podcast",
    "create_podcast_with_random_texts",

    # Übersetzung
    "translate_audio",
]
