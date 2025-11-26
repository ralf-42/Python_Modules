"""
audio_lib - Konfiguration

Konfigurationsklasse für die Audio-Bibliothek.
Enthält API-Keys, Modelle und Standard-Einstellungen.
"""

import os
from typing import Dict, Optional


class AudioConfig:
    """
    Konfigurationsklasse für audio_lib.

    Attribute:
        openai_api_key (str): OpenAI API-Key
        stt_model (str): Speech-to-Text Modell (Standard: whisper-1)
        tts_model (str): Text-to-Speech Modell (Standard: tts-1)
        llm_model (str): LLM Modell für Analysen (Standard: gpt-4o-mini)
        default_voice (str): Standard-Stimme für TTS (Standard: alloy)
        available_voices (list): Liste verfügbarer Stimmen
        default_temperature (float): Standard-Temperature für LLM (Standard: 0.2)
    """

    # Verfügbare Stimmen mit Beschreibungen
    VOICES = {
        'alloy': 'neutral',
        'echo': 'jugendlich',
        'fable': 'männlich',
        'onyx': 'tiefe männliche Stimme',
        'nova': 'weiblich',
        'shimmer': 'warme weibliche Stimme'
    }

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        stt_model: str = "whisper-1",
        tts_model: str = "tts-1",
        llm_model: str = "gpt-4o-mini",
        default_voice: str = "alloy",
        default_temperature: float = 0.2
    ):
        """
        Initialisiert die AudioConfig.

        Args:
            openai_api_key: OpenAI API-Key (optional, wird sonst aus Umgebungsvariable gelesen)
            stt_model: Speech-to-Text Modell
            tts_model: Text-to-Speech Modell (tts-1 oder tts-1-hd)
            llm_model: LLM Modell für Analysen
            default_voice: Standard-Stimme
            default_temperature: Temperature für LLM-Aufrufe
        """
        # API-Key aus Parameter oder Umgebungsvariable
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')

        if not self.openai_api_key:
            raise ValueError(
                "OpenAI API-Key nicht gefunden. Bitte als Parameter übergeben "
                "oder in Umgebungsvariable OPENAI_API_KEY setzen."
            )

        # Modell-Konfiguration
        self.stt_model = stt_model
        self.tts_model = tts_model
        self.llm_model = llm_model

        # Stimmen-Konfiguration
        if default_voice not in self.VOICES:
            raise ValueError(
                f"Ungültige Stimme '{default_voice}'. "
                f"Verfügbar: {', '.join(self.VOICES.keys())}"
            )
        self.default_voice = default_voice
        self.available_voices = list(self.VOICES.keys())

        # LLM-Einstellungen
        self.default_temperature = default_temperature

    def get_voice_description(self, voice: str) -> str:
        """
        Gibt die Beschreibung einer Stimme zurück.

        Args:
            voice: Name der Stimme

        Returns:
            Beschreibung der Stimme
        """
        return self.VOICES.get(voice, "Unbekannte Stimme")

    def validate_voice(self, voice: str) -> bool:
        """
        Überprüft, ob eine Stimme verfügbar ist.

        Args:
            voice: Name der Stimme

        Returns:
            True wenn verfügbar, sonst False
        """
        return voice in self.VOICES

    def __repr__(self):
        return (
            f"AudioConfig(stt_model='{self.stt_model}', "
            f"tts_model='{self.tts_model}', "
            f"llm_model='{self.llm_model}', "
            f"default_voice='{self.default_voice}')"
        )
