# audio_lib

High-Level Python-Bibliothek für **Text-to-Speech (TTS)** und **Speech-to-Text (STT)** mit OpenAI's Whisper und TTS-Modellen.

## Features

- 🎙️ **Speech-to-Text (STT)** - Transkription von Audiodateien mit Whisper
- 🔊 **Text-to-Speech (TTS)** - Natürliche Sprachausgabe mit 6 verschiedenen Stimmen
- 👥 **Multi-Speaker Audio** - Audio aus Textdateien mit verschiedenen Sprechern
- 📊 **Audio-Analyse** - Sentiment- und Sprachanalyse
- 🎧 **Podcast-Generierung** - Automatische Podcast-Erstellung
- 🌍 **Audio-Übersetzung** - Transkription → Übersetzung → TTS
- 🔧 **Audio-Utilities** - Kombinieren, Pausen einfügen, etc.

## Installation

```bash
# Bibliothek im Python-Path verfügbar machen
export PYTHONPATH="${PYTHONPATH}:/pfad/zu/GenAI/Python_Modules"
```

### Abhängigkeiten

```bash
pip install openai pydub scipy
```

Zusätzlich wird `ffmpeg` für pydub benötigt:

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download von https://ffmpeg.org/download.html
```

## Schnellstart

```python
from audio_lib import AudioConfig, transcribe_audio, text_to_speech

# Konfiguration
config = AudioConfig()  # Nutzt OPENAI_API_KEY aus Umgebungsvariable

# Speech-to-Text
text = transcribe_audio('interview.mp3', config)
print(text)

# Text-to-Speech
text_to_speech("Hallo Welt!", "output.mp3", config, voice='nova')
```

## Konfiguration

```python
from audio_lib import AudioConfig

# Mit Umgebungsvariable OPENAI_API_KEY
config = AudioConfig()

# Oder explizit
config = AudioConfig(
    openai_api_key="sk-...",
    stt_model="whisper-1",
    tts_model="tts-1",  # oder "tts-1-hd"
    llm_model="gpt-4o-mini",
    default_voice="alloy",
    default_temperature=0.2
)

# Verfügbare Stimmen
print(config.available_voices)
# ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']

# Stimmen-Beschreibungen
for voice in config.available_voices:
    desc = config.get_voice_description(voice)
    print(f"{voice}: {desc}")
```

## Beispiele

### 1. Speech-to-Text (STT)

#### Einfache Transkription

```python
from audio_lib import transcribe_audio, AudioConfig

config = AudioConfig()
text = transcribe_audio('podcast.mp3', config)
print(text)
```

#### Transkription mit Zeitstempeln

```python
from audio_lib import transcribe_with_timestamps, AudioConfig

config = AudioConfig()
segments = transcribe_with_timestamps('interview.mp3', config)

for seg in segments:
    print(f"{seg['start']:.2f}s - {seg['end']:.2f}s: {seg['text']}")
```

#### Transkription mit Sprachcode

```python
text = transcribe_audio('audio.mp3', config, language='de')
```

### 2. Text-to-Speech (TTS)

#### Einfaches TTS

```python
from audio_lib import text_to_speech, AudioConfig

config = AudioConfig()

text = "Willkommen zur GenAI Vorlesung!"
text_to_speech(text, 'begruessung.mp3', config, voice='nova')
```

#### Verschiedene Stimmen testen

```python
text = "Dies ist ein Test der verschiedenen Stimmen."
voices = ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']

for voice in voices:
    output_file = f'test_{voice}.mp3'
    text_to_speech(text, output_file, config, voice=voice)
    print(f"Erstellt: {output_file}")
```

### 3. Multi-Speaker Audio ⭐

Dies ist die Hauptfunktion für Dialog-Erstellung aus Textdateien!

#### Textdatei erstellen: `dialog.txt`

```
Sprecher 1: Guten Tag! Willkommen zu unserem Interview.
Sprecher 2: Vielen Dank! Ich freue mich hier zu sein.
Sprecher 1: Erzählen Sie uns etwas über Ihre Arbeit mit KI.
Sprecher 2: Ich beschäftige mich hauptsächlich mit generativen Modellen.
Sprecher 1: Das ist sehr spannend! Wie sehen Sie die Zukunft?
Sprecher 2: Ich denke, KI wird in allen Bereichen eine große Rolle spielen.
```

#### Audio erstellen

```python
from audio_lib import create_multi_speaker_audio, AudioConfig

config = AudioConfig()

# Stimmen für Sprecher definieren
speaker_voices = {
    1: 'alloy',   # Sprecher 1 = alloy (neutral)
    2: 'nova',    # Sprecher 2 = nova (weiblich)
}

# Audio generieren
create_multi_speaker_audio(
    text_file='dialog.txt',
    speaker_voices=speaker_voices,
    output_file='dialog.mp3',
    config=config,
    pause_duration=500  # 500ms Pause zwischen Sprechern
)

print("Dialog erstellt: dialog.mp3")
```

#### Komplexeres Beispiel mit 3 Sprechern

```python
# 3-Personen Diskussion
speaker_voices = {
    1: 'alloy',   # Moderator
    2: 'nova',    # Expertin
    3: 'onyx',    # Experte
}

create_multi_speaker_audio(
    'diskussion.txt',
    speaker_voices,
    'diskussion.mp3',
    config,
    pause_duration=700
)
```

### 4. Audio-Analyse

#### Zusammenfassung erstellen

```python
from audio_lib import create_audio_summary, AudioConfig

config = AudioConfig()

# Nur Text-Zusammenfassung
result = create_audio_summary('vortrag.mp3', config, max_sentences=3)
print("Zusammenfassung:", result['summary'])
print("Original:", result['transcript'])

# Mit Audio-Ausgabe der Zusammenfassung
result = create_audio_summary(
    'vortrag.mp3',
    config,
    max_sentences=4,
    output_audio='zusammenfassung.mp3',
    voice='shimmer'
)
print("Zusammenfassung als Audio gespeichert:", result['audio_file'])
```

#### Sentiment-Analyse

```python
from audio_lib import analyze_audio_sentiment, AudioConfig

config = AudioConfig()

result = analyze_audio_sentiment('rede.mp3', config)
print("Transkript:", result['transcript'])
print("\nAnalyse:")
print(result['analysis'])
```

### 5. Podcast-Generierung

#### Standard-Podcast

```python
from audio_lib import create_podcast, AudioConfig

config = AudioConfig()

intro = "Willkommen zu unserem Podcast über künstliche Intelligenz!"
main = """
Heute sprechen wir über generative Modelle. Diese Modelle können
Text, Bilder und sogar Audio erzeugen. Die Entwicklung ist rasant
und eröffnet völlig neue Möglichkeiten.
"""
outro = "Vielen Dank fürs Zuhören! Bis zur nächsten Folge!"

create_podcast(
    intro_text=intro,
    main_content=main,
    outro_text=outro,
    output_file='podcast_episode1.mp3',
    config=config,
    voices={
        'intro': 'nova',
        'main': 'alloy',
        'outro': 'nova'
    },
    pause_duration=1000
)
```

#### Podcast mit zufälligen Intro/Outro

```python
from audio_lib import create_podcast_with_random_texts, AudioConfig

config = AudioConfig()

main_content = """
In der heutigen Folge diskutieren wir die neuesten Entwicklungen
im Bereich der Large Language Models und ihre Anwendungen.
"""

create_podcast_with_random_texts(
    main_content=main_content,
    output_file='podcast_random.mp3',
    config=config,
    voices={'main': 'alloy'}
)
```

### 6. Audio-Übersetzung

```python
from audio_lib import translate_audio, AudioConfig

config = AudioConfig()

# Deutsch → Englisch
result = translate_audio(
    audio_file='deutsch.mp3',
    target_language='Englisch',
    output_file='english.mp3',
    config=config,
    voice='alloy'
)

print("Original:", result['original_transcript'])
print("Übersetzung:", result['translation'])
print("Audio:", result['audio_file'])

# Deutsch → Spanisch
result = translate_audio(
    'deutsch.mp3',
    'Spanisch',
    'spanish.mp3',
    config,
    voice='shimmer'
)
```

### 7. Audio kombinieren

```python
from audio_lib import combine_audio_files

files = [
    'intro.mp3',
    'hauptteil.mp3',
    'interview.mp3',
    'outro.mp3'
]

combine_audio_files(
    audio_files=files,
    output_file='kompletter_podcast.mp3',
    pause_duration=1500  # 1.5 Sekunden Pause
)
```

## API-Referenz

### Konfiguration

#### `AudioConfig`

```python
AudioConfig(
    openai_api_key: Optional[str] = None,
    stt_model: str = "whisper-1",
    tts_model: str = "tts-1",
    llm_model: str = "gpt-4o-mini",
    default_voice: str = "alloy",
    default_temperature: float = 0.2
)
```

### STT-Funktionen

#### `transcribe_audio()`

```python
transcribe_audio(
    audio_file: str,
    config: AudioConfig,
    language: Optional[str] = None
) -> str
```

#### `transcribe_with_timestamps()`

```python
transcribe_with_timestamps(
    audio_file: str,
    config: AudioConfig
) -> List[Dict[str, Union[float, str]]]
```

### TTS-Funktionen

#### `text_to_speech()`

```python
text_to_speech(
    text: str,
    output_file: str,
    config: AudioConfig,
    voice: Optional[str] = None,
    model: Optional[str] = None
) -> str
```

#### `create_multi_speaker_audio()`

```python
create_multi_speaker_audio(
    text_file: str,
    speaker_voices: Dict[int, str],
    output_file: str,
    config: AudioConfig,
    pause_duration: int = 500
) -> str
```

**Textdatei-Format:**
```
Sprecher 1: Text von Sprecher 1
Sprecher 2: Text von Sprecher 2
Sprecher 1: Weitere Aussage
```

### Analyse-Funktionen

#### `create_audio_summary()`

```python
create_audio_summary(
    audio_file: str,
    config: AudioConfig,
    max_sentences: int = 4,
    output_audio: Optional[str] = None,
    voice: Optional[str] = None
) -> Dict[str, str]
```

#### `analyze_audio_sentiment()`

```python
analyze_audio_sentiment(
    audio_file: str,
    config: AudioConfig
) -> Dict[str, str]
```

### Podcast-Funktionen

#### `create_podcast()`

```python
create_podcast(
    intro_text: str,
    main_content: str,
    outro_text: str,
    output_file: str,
    config: AudioConfig,
    voices: Optional[Dict[str, str]] = None,
    pause_duration: int = 1000
) -> str
```

#### `create_podcast_with_random_texts()`

```python
create_podcast_with_random_texts(
    main_content: str,
    output_file: str,
    config: AudioConfig,
    voices: Optional[Dict[str, str]] = None,
    pause_duration: int = 1000
) -> str
```

### Übersetzung

#### `translate_audio()`

```python
translate_audio(
    audio_file: str,
    target_language: str,
    output_file: str,
    config: AudioConfig,
    voice: Optional[str] = None
) -> Dict[str, str]
```

### Utilities

#### `combine_audio_files()`

```python
combine_audio_files(
    audio_files: List[str],
    output_file: str,
    pause_duration: int = 1000
) -> str
```

## Verfügbare Stimmen

| Stimme | Beschreibung |
|--------|--------------|
| `alloy` | neutral |
| `echo` | jugendlich |
| `fable` | männlich |
| `onyx` | tiefe männliche Stimme |
| `nova` | weiblich |
| `shimmer` | warme weibliche Stimme |

## Tipps & Best Practices

### Multi-Speaker Audio

1. **Stimmen-Auswahl**: Wählen Sie deutlich unterscheidbare Stimmen
   ```python
   # Gut: Klarer Kontrast
   voices = {1: 'alloy', 2: 'nova', 3: 'onyx'}
   ```

2. **Pausen anpassen**: Längere Pausen für natürlicheren Dialog
   ```python
   pause_duration=700  # 700ms für natürlichere Gespräche
   ```

3. **Text strukturieren**: Kurze, klare Sätze pro Sprecher

### Audio-Qualität

- Nutzen Sie `tts-1-hd` für bessere Qualität (höhere Latenz)
  ```python
  config = AudioConfig(tts_model="tts-1-hd")
  ```

### Performance

- Bei vielen Audio-Dateien: Temporäre Dateien zwischenspeichern
- Große Transkriptionen: Dateien vorher in kleinere Segmente teilen

## Fehlerbehandlung

```python
from audio_lib import AudioConfig, transcribe_audio

try:
    config = AudioConfig()
    text = transcribe_audio('audio.mp3', config)
except FileNotFoundError as e:
    print(f"Datei nicht gefunden: {e}")
except ValueError as e:
    print(f"Fehler in der Konfiguration: {e}")
except Exception as e:
    print(f"Unerwarteter Fehler: {e}")
```

## Anwendungsbeispiele

### 1. Automatischer Podcast aus Blog-Post

```python
from audio_lib import create_podcast_with_random_texts, AudioConfig

config = AudioConfig()

with open('blogpost.txt', 'r') as f:
    content = f.read()

create_podcast_with_random_texts(
    content,
    'blog_podcast.mp3',
    config,
    voices={'main': 'nova'}
)
```

### 2. Interview-Transkription mit Zeitstempeln

```python
from audio_lib import transcribe_with_timestamps, AudioConfig

config = AudioConfig()
segments = transcribe_with_timestamps('interview.mp3', config)

# Als Untertitel-Datei speichern
with open('untertitel.srt', 'w') as f:
    for i, seg in enumerate(segments, 1):
        f.write(f"{i}\n")
        f.write(f"{seg['start']:.2f} --> {seg['end']:.2f}\n")
        f.write(f"{seg['text']}\n\n")
```

### 3. Meeting-Analyse

```python
from audio_lib import analyze_audio_sentiment, create_audio_summary, AudioConfig

config = AudioConfig()

# Zusammenfassung
summary = create_audio_summary('meeting.mp3', config, max_sentences=5)
print("Meeting Summary:", summary['summary'])

# Sentiment
sentiment = analyze_audio_sentiment('meeting.mp3', config)
print("Stimmung:", sentiment['analysis'])
```

## Lizenz

MIT License - Copyright (c) 2025 Ralf

## Autor

Ralf - GenAI Kurs 2025

Basiert auf dem Notebook M15_Multimodal_Audio.ipynb
