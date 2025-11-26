"""
Beispiel-Skript: audio_lib Verwendung

Dieses Skript zeigt die grundlegende Verwendung der audio_lib Bibliothek.
"""

import sys
sys.path.append('../Python_Modules')

from audio_lib import (
    AudioConfig,
    create_multi_speaker_audio,
    text_to_speech,
    transcribe_audio,
    create_podcast_with_random_texts,
)


def main():
    """Hauptfunktion mit Beispielen."""

    # ========================================================================
    # 1. Konfiguration
    # ========================================================================
    print("=" * 70)
    print("1. AudioConfig initialisieren")
    print("=" * 70)

    config = AudioConfig()
    print(f"✅ Konfiguration erstellt: {config}")
    print()

    # ========================================================================
    # 2. Einfaches Text-to-Speech
    # ========================================================================
    print("=" * 70)
    print("2. Einfaches Text-to-Speech")
    print("=" * 70)

    text = "Willkommen zur Demonstration der audio_lib Bibliothek!"
    output_file = text_to_speech(text, 'beispiel_simple.mp3', config, voice='nova')

    print(f"✅ Audio erstellt: {output_file}")
    print()

    # ========================================================================
    # 3. Multi-Speaker Audio erstellen
    # ========================================================================
    print("=" * 70)
    print("3. Multi-Speaker Audio aus Textdatei")
    print("=" * 70)

    # Stimmen für Sprecher definieren
    speaker_voices = {
        1: 'alloy',   # Sprecher 1
        2: 'nova',    # Sprecher 2
    }

    # Prüfen ob Textdatei existiert
    try:
        output_file = create_multi_speaker_audio(
            text_file='beispiel_interview_ki.txt',
            speaker_voices=speaker_voices,
            output_file='beispiel_interview.mp3',
            config=config,
            pause_duration=600
        )
        print(f"✅ Multi-Speaker Audio erstellt: {output_file}")
    except FileNotFoundError as e:
        print(f"⚠️  Textdatei nicht gefunden: {e}")
        print("   Tipp: Stellen Sie sicher, dass 'beispiel_interview_ki.txt' im Verzeichnis liegt.")

    print()

    # ========================================================================
    # 4. Podcast mit zufälligen Intro/Outro
    # ========================================================================
    print("=" * 70)
    print("4. Podcast-Generierung")
    print("=" * 70)

    main_content = """
    In der heutigen Episode sprechen wir über die audio_lib Bibliothek.
    Diese Bibliothek macht es einfach, Text in natürliche Sprache umzuwandeln
    und mehrere Sprecher in einem Dialog zu kombinieren.
    Die Anwendungsmöglichkeiten sind vielfältig, von Hörbüchern
    bis hin zu automatisierten Podcasts.
    """

    output_file = create_podcast_with_random_texts(
        main_content=main_content,
        output_file='beispiel_podcast.mp3',
        config=config,
        voices={'main': 'alloy'},
        pause_duration=1000
    )

    print(f"✅ Podcast erstellt: {output_file}")
    print()

    # ========================================================================
    # 5. Speech-to-Text (Transkription)
    # ========================================================================
    print("=" * 70)
    print("5. Speech-to-Text (Audio transkribieren)")
    print("=" * 70)

    # Transkribiere eine der erstellten Dateien
    try:
        transcript = transcribe_audio('beispiel_simple.mp3', config)
        print(f"📄 Transkript: {transcript}")
    except FileNotFoundError as e:
        print(f"⚠️  Audio-Datei nicht gefunden: {e}")

    print()

    # ========================================================================
    # Zusammenfassung
    # ========================================================================
    print("=" * 70)
    print("✅ Fertig! Erstellte Dateien:")
    print("=" * 70)
    print("  • beispiel_simple.mp3 - Einfaches TTS")
    print("  • beispiel_interview.mp3 - Multi-Speaker Dialog (falls Textdatei vorhanden)")
    print("  • beispiel_podcast.mp3 - Podcast mit zufälligen Texten")
    print()
    print("Tipp: Verwenden Sie einen Audio-Player, um die Dateien anzuhören!")
    print()


if __name__ == "__main__":
    main()
