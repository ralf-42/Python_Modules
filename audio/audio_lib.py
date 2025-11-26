"""
audio_lib - High-Level Audio-Bibliothek für TTS und STT

Diese Bibliothek bietet einfache High-Level-Funktionen für:
- Speech-to-Text (STT) mit OpenAI Whisper
- Text-to-Speech (TTS) mit OpenAI TTS
- Multi-Speaker Audio-Generierung
- Audio-Analyse und Sentiment-Analyse
- Podcast-Generierung
- Audio-Übersetzung
- Audio-Kombinierung

Autor: Ralf
Erstellt: 2025
"""

import os
import re
import random
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

from openai import OpenAI
from pydub import AudioSegment


# ============================================================================
# HILFSFUNKTIONEN
# ============================================================================

def _ensure_client(config) -> OpenAI:
    """
    Erstellt einen OpenAI-Client mit dem API-Key aus der Config.

    Args:
        config: AudioConfig-Instanz

    Returns:
        OpenAI Client
    """
    return OpenAI(api_key=config.openai_api_key)


def _validate_file_exists(file_path: str) -> Path:
    """
    Überprüft, ob eine Datei existiert.

    Args:
        file_path: Pfad zur Datei

    Returns:
        Path-Objekt

    Raises:
        FileNotFoundError: Wenn Datei nicht existiert
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {file_path}")
    return path


def _parse_speaker_line(line: str) -> Optional[Tuple[int, str]]:
    """
    Parst eine Zeile mit Sprecher-Format: "Sprecher X: Text"

    Args:
        line: Textzeile

    Returns:
        Tuple (Sprecher-Nummer, Text) oder None wenn kein Match
    """
    # Pattern: "Sprecher 1:", "Sprecher 2:", etc.
    pattern = r'^Sprecher\s+(\d+)\s*:\s*(.+)$'
    match = re.match(pattern, line.strip(), re.IGNORECASE)

    if match:
        speaker_num = int(match.group(1))
        text = match.group(2).strip()
        return speaker_num, text

    return None


# ============================================================================
# STT FUNKTIONEN (Speech-to-Text)
# ============================================================================

def transcribe_audio(
    audio_file: str,
    config,
    language: Optional[str] = None
) -> str:
    """
    Transkribiert eine Audiodatei zu Text.

    Args:
        audio_file: Pfad zur Audiodatei
        config: AudioConfig-Instanz
        language: Optionale Sprachcode (z.B. 'de', 'en')

    Returns:
        Transkribierter Text

    Example:
        >>> from audio_lib import transcribe_audio, AudioConfig
        >>> config = AudioConfig()
        >>> text = transcribe_audio('interview.mp3', config)
    """
    path = _validate_file_exists(audio_file)
    client = _ensure_client(config)

    with open(path, "rb") as audio:
        kwargs = {
            "model": config.stt_model,
            "file": audio
        }
        if language:
            kwargs["language"] = language

        response = client.audio.transcriptions.create(**kwargs)

    return response.text


def transcribe_with_timestamps(
    audio_file: str,
    config
) -> List[Dict[str, Union[float, str]]]:
    """
    Transkribiert eine Audiodatei mit Zeitstempeln.

    Args:
        audio_file: Pfad zur Audiodatei
        config: AudioConfig-Instanz

    Returns:
        Liste von Segmenten mit start, end, text

    Example:
        >>> segments = transcribe_with_timestamps('podcast.mp3', config)
        >>> for seg in segments:
        ...     print(f"{seg['start']:.2f}s - {seg['end']:.2f}s: {seg['text']}")
    """
    path = _validate_file_exists(audio_file)
    client = _ensure_client(config)

    with open(path, "rb") as audio:
        response = client.audio.transcriptions.create(
            model=config.stt_model,
            file=audio,
            response_format="verbose_json"
        )

    # Segmente extrahieren
    segments = []
    for seg in response.segments:
        segments.append({
            'start': seg.start,
            'end': seg.end,
            'text': seg.text
        })

    return segments


# ============================================================================
# TTS FUNKTIONEN (Text-to-Speech)
# ============================================================================

def text_to_speech(
    text: str,
    output_file: str,
    config,
    voice: Optional[str] = None,
    model: Optional[str] = None
) -> str:
    """
    Wandelt Text in Sprache um und speichert als MP3.

    Args:
        text: Text der gesprochen werden soll
        output_file: Pfad zur Ausgabe-Audiodatei
        config: AudioConfig-Instanz
        voice: Stimme (Standard: aus config)
        model: TTS-Modell (Standard: aus config)

    Returns:
        Pfad zur erstellten Audiodatei

    Example:
        >>> text_to_speech("Hallo Welt", "output.mp3", config, voice="nova")
    """
    client = _ensure_client(config)

    # Standard-Werte aus Config
    voice = voice or config.default_voice
    model = model or config.tts_model

    # Validierung
    if not config.validate_voice(voice):
        raise ValueError(
            f"Ungültige Stimme '{voice}'. "
            f"Verfügbar: {', '.join(config.available_voices)}"
        )

    # TTS-Aufruf
    response = client.audio.speech.create(
        model=model,
        voice=voice,
        input=text
    )

    # Speichern
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(response.content)

    return str(output_path)


def create_multi_speaker_audio(
    text_file: str,
    speaker_voices: Dict[int, str],
    output_file: str,
    config,
    pause_duration: int = 500
) -> str:
    """
    Erstellt eine Audio-Datei aus einer Text-Datei mit mehreren Sprechern.

    Format der Text-Datei:
        Sprecher 1: Text von Sprecher 1
        Sprecher 2: Text von Sprecher 2
        Sprecher 1: Weitere Aussage von Sprecher 1

    Args:
        text_file: Pfad zur Text-Datei
        speaker_voices: Dict mit Sprecher-Nummer → Stimme (z.B. {1: 'alloy', 2: 'nova'})
        output_file: Pfad zur Ausgabe-Audiodatei
        config: AudioConfig-Instanz
        pause_duration: Pause zwischen Sprechern in Millisekunden (Standard: 500ms)

    Returns:
        Pfad zur erstellten Audiodatei

    Example:
        >>> voices = {1: 'alloy', 2: 'nova', 3: 'onyx'}
        >>> create_multi_speaker_audio('dialog.txt', voices, 'dialog.mp3', config)
    """
    path = _validate_file_exists(text_file)

    # Text-Datei lesen und parsen
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Sprecher-Segmente extrahieren
    segments = []
    for line_num, line in enumerate(lines, 1):
        parsed = _parse_speaker_line(line)
        if parsed:
            speaker_num, text = parsed

            # Prüfen ob Stimme für Sprecher definiert
            if speaker_num not in speaker_voices:
                raise ValueError(
                    f"Keine Stimme für Sprecher {speaker_num} definiert. "
                    f"Zeile {line_num}: {line.strip()}"
                )

            segments.append({
                'speaker': speaker_num,
                'voice': speaker_voices[speaker_num],
                'text': text
            })

    if not segments:
        raise ValueError(
            f"Keine Sprecher-Segmente in {text_file} gefunden. "
            "Format: 'Sprecher X: Text'"
        )

    # Temporäre Audiodateien erstellen
    temp_files = []
    client = _ensure_client(config)

    for i, segment in enumerate(segments):
        temp_file = f"temp_speaker_{segment['speaker']}_{i}.mp3"

        response = client.audio.speech.create(
            model=config.tts_model,
            voice=segment['voice'],
            input=segment['text']
        )

        with open(temp_file, "wb") as f:
            f.write(response.content)

        temp_files.append(temp_file)

    # Audio-Segmente kombinieren
    combined = AudioSegment.empty()
    pause = AudioSegment.silent(duration=pause_duration)

    for i, temp_file in enumerate(temp_files):
        audio = AudioSegment.from_file(temp_file)
        combined += audio

        # Pause hinzufügen (außer beim letzten Segment)
        if i < len(temp_files) - 1:
            combined += pause

    # Ausgabe speichern
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.export(output_path, format="mp3")

    # Temporäre Dateien löschen
    for temp_file in temp_files:
        os.remove(temp_file)

    return str(output_path)


# ============================================================================
# UTILITIES
# ============================================================================

def combine_audio_files(
    audio_files: List[str],
    output_file: str,
    pause_duration: int = 1000
) -> str:
    """
    Kombiniert mehrere Audiodateien zu einer einzigen.

    Args:
        audio_files: Liste von Pfaden zu Audiodateien
        output_file: Pfad zur kombinierten Ausgabe-Datei
        pause_duration: Pause zwischen Dateien in Millisekunden (Standard: 1000ms)

    Returns:
        Pfad zur kombinierten Audiodatei

    Example:
        >>> files = ['intro.mp3', 'hauptteil.mp3', 'outro.mp3']
        >>> combine_audio_files(files, 'podcast.mp3', pause_duration=500)
    """
    if not audio_files:
        raise ValueError("Keine Audiodateien angegeben")

    # Alle Dateien validieren
    for f in audio_files:
        _validate_file_exists(f)

    # Kombinieren
    combined = AudioSegment.empty()
    pause = AudioSegment.silent(duration=pause_duration)

    for i, audio_file in enumerate(audio_files):
        audio = AudioSegment.from_file(audio_file)
        combined += audio

        # Pause hinzufügen (außer beim letzten)
        if i < len(audio_files) - 1:
            combined += pause

    # Speichern
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.export(output_path, format="mp3")

    return str(output_path)


# ============================================================================
# AUDIO-ANALYSE
# ============================================================================

def create_audio_summary(
    audio_file: str,
    config,
    max_sentences: int = 4,
    output_audio: Optional[str] = None,
    voice: Optional[str] = None
) -> Dict[str, str]:
    """
    Erstellt eine Zusammenfassung einer Audiodatei.

    Workflow: Audio → STT → LLM Summary → (optional) TTS

    Args:
        audio_file: Pfad zur Audiodatei
        config: AudioConfig-Instanz
        max_sentences: Maximale Anzahl Sätze in der Zusammenfassung
        output_audio: Optional: Pfad für Audio-Ausgabe der Zusammenfassung
        voice: Stimme für Audio-Ausgabe (Standard: aus config)

    Returns:
        Dict mit 'transcript' und 'summary' (und optional 'audio_file')

    Example:
        >>> result = create_audio_summary('interview.mp3', config, output_audio='summary.mp3')
        >>> print(result['summary'])
    """
    # Transkription
    transcript = transcribe_audio(audio_file, config)

    # LLM-Summary
    client = _ensure_client(config)
    prompt = f"""
Erstelle eine kurze Zusammenfassung des folgenden Textes.
Die Zusammenfassung sollte maximal {max_sentences} Sätze umfassen und die Hauptpunkte klar darstellen.

### Transkript:
{transcript}
"""

    response = client.chat.completions.create(
        model=config.llm_model,
        messages=[
            {"role": "system", "content": "Du bist ein Experte für Textzusammenfassungen."},
            {"role": "user", "content": prompt}
        ],
        temperature=config.default_temperature
    )

    summary = response.choices[0].message.content

    result = {
        'transcript': transcript,
        'summary': summary
    }

    # Optional: TTS der Zusammenfassung
    if output_audio:
        audio_path = text_to_speech(summary, output_audio, config, voice)
        result['audio_file'] = audio_path

    return result


def analyze_audio_sentiment(
    audio_file: str,
    config
) -> Dict[str, str]:
    """
    Analysiert eine Audiodatei auf Stimmung, Tonfall und weitere Merkmale.

    Args:
        audio_file: Pfad zur Audiodatei
        config: AudioConfig-Instanz

    Returns:
        Dict mit 'transcript' und 'analysis'

    Example:
        >>> result = analyze_audio_sentiment('speech.mp3', config)
        >>> print(result['analysis'])
    """
    # Transkription
    transcript = transcribe_audio(audio_file, config)

    # Analyse-Prompt (basierend auf M15 Notebook)
    client = _ensure_client(config)
    prompt = f"""
Analysiere den folgenden transkribierten Text umfassend:

🔹 Inhaltsanalyse:
    - Themen und Argumentationsstruktur
    - Sprachstil (Alltagssprache vs. Fachsprache, formell vs. informell)
    - Rhetorische Mittel
    - Satzstruktur & Wortwahl
    - Tonfall und Haltung

🔹 Stimmungsanalyse:
    - Stimmung: Positiv 😊, Neutral 😐, Negativ 😞
    - Begründe Deine Einschätzung

🔹 Wirkung auf Zuhörer:
    - Verständlichkeit
    - Authentizität
    - Emotionale Ansprache
    - Zielgruppenorientierung

🔹 Weitere Aspekte:
    - Ethische Aspekte
    - Versteckte Botschaften
    - Weitere interessante Punkte

Vermeide Formulierungen wie "könnte" oder "sollte". Nimm konkret Stellung.

### Text:
{transcript}
"""

    response = client.chat.completions.create(
        model=config.llm_model,
        messages=[
            {"role": "system", "content": "Du bist ein Experte für Stimmungs- und Sprachanalysen."},
            {"role": "user", "content": prompt}
        ],
        temperature=config.default_temperature
    )

    analysis = response.choices[0].message.content

    return {
        'transcript': transcript,
        'analysis': analysis
    }


# ============================================================================
# PODCAST-GENERIERUNG
# ============================================================================

def create_podcast(
    intro_text: str,
    main_content: str,
    outro_text: str,
    output_file: str,
    config,
    voices: Optional[Dict[str, str]] = None,
    pause_duration: int = 1000
) -> str:
    """
    Erstellt einen kompletten Podcast mit Intro, Hauptteil und Outro.

    Args:
        intro_text: Einleitungstext
        main_content: Hauptinhalt
        outro_text: Abschlusstext
        output_file: Pfad zur Ausgabe-Datei
        config: AudioConfig-Instanz
        voices: Optional Dict mit 'intro', 'main', 'outro' → Stimme
        pause_duration: Pause zwischen Segmenten in Millisekunden

    Returns:
        Pfad zur erstellten Podcast-Datei

    Example:
        >>> create_podcast(
        ...     "Willkommen zum Podcast!",
        ...     "Heute sprechen wir über KI...",
        ...     "Danke fürs Zuhören!",
        ...     "podcast.mp3",
        ...     config,
        ...     voices={'intro': 'nova', 'main': 'alloy', 'outro': 'nova'}
        ... )
    """
    # Standard-Stimmen
    if voices is None:
        voices = {
            'intro': config.default_voice,
            'main': config.default_voice,
            'outro': config.default_voice
        }

    # Temporäre Dateien erstellen
    temp_files = []

    segments = [
        ('intro', intro_text, voices.get('intro', config.default_voice)),
        ('main', main_content, voices.get('main', config.default_voice)),
        ('outro', outro_text, voices.get('outro', config.default_voice))
    ]

    for name, text, voice in segments:
        temp_file = f"temp_podcast_{name}.mp3"
        text_to_speech(text, temp_file, config, voice)
        temp_files.append(temp_file)

    # Kombinieren
    result = combine_audio_files(temp_files, output_file, pause_duration)

    # Temporäre Dateien löschen
    for temp_file in temp_files:
        os.remove(temp_file)

    return result


def create_podcast_with_random_texts(
    main_content: str,
    output_file: str,
    config,
    voices: Optional[Dict[str, str]] = None,
    pause_duration: int = 1000
) -> str:
    """
    Erstellt einen Podcast mit zufälligen Intro/Outro-Texten.

    Args:
        main_content: Hauptinhalt
        output_file: Pfad zur Ausgabe-Datei
        config: AudioConfig-Instanz
        voices: Optional Dict mit 'intro', 'main', 'outro' → Stimme
        pause_duration: Pause zwischen Segmenten in Millisekunden

    Returns:
        Pfad zur erstellten Podcast-Datei
    """
    # Zufällige Intros
    intros = [
        "Schön, dass du dabei bist! In diesem Podcast sprechen wir über Themen, die uns wirklich beschäftigen.",
        "Hallo und herzlich willkommen! Heute geht's um ein Thema, das aktueller nicht sein könnte.",
        "Willkommen beim Podcast über die großen und kleinen Fragen unseres Alltags."
    ]

    # Zufällige Outros
    outros = [
        "Puh, ganz schön viel Stoff heute! Nimm dir Zeit, das alles wirken zu lassen. Bis zum nächsten Mal!",
        "Krass! Das waren viele spannende Einblicke – schön, dass du dabei warst! Bis bald!",
        "Wow! Danke, dass du heute dabei warst. Wir freuen uns schon auf die nächste Folge mit dir!"
    ]

    intro = random.choice(intros)
    outro = random.choice(outros)

    return create_podcast(intro, main_content, outro, output_file, config, voices, pause_duration)


# ============================================================================
# AUDIO-ÜBERSETZUNG
# ============================================================================

def translate_audio(
    audio_file: str,
    target_language: str,
    output_file: str,
    config,
    voice: Optional[str] = None
) -> Dict[str, str]:
    """
    Übersetzt eine Audiodatei in eine andere Sprache.

    Workflow: Audio → STT → LLM Translation → TTS

    Args:
        audio_file: Pfad zur Audiodatei
        target_language: Zielsprache (z.B. 'Englisch', 'Spanisch', 'Französisch')
        output_file: Pfad zur übersetzten Audio-Ausgabe
        config: AudioConfig-Instanz
        voice: Stimme für die Ausgabe (Standard: aus config)

    Returns:
        Dict mit 'original_transcript', 'translation', 'audio_file'

    Example:
        >>> result = translate_audio('deutsch.mp3', 'Englisch', 'english.mp3', config, voice='alloy')
        >>> print(result['translation'])
    """
    # Transkription
    original_transcript = transcribe_audio(audio_file, config)

    # Übersetzung via LLM
    client = _ensure_client(config)
    prompt = f"""
Übersetze den folgenden Text ins {target_language}.
Achte auf natürliche Formulierungen und den Kontext.

### Original-Text:
{original_transcript}
"""

    response = client.chat.completions.create(
        model=config.llm_model,
        messages=[
            {"role": "system", "content": f"Du bist ein professioneller Übersetzer ins {target_language}."},
            {"role": "user", "content": prompt}
        ],
        temperature=config.default_temperature
    )

    translation = response.choices[0].message.content

    # TTS der Übersetzung
    audio_path = text_to_speech(translation, output_file, config, voice)

    return {
        'original_transcript': original_transcript,
        'translation': translation,
        'audio_file': audio_path
    }
