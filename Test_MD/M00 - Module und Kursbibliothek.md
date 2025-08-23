
![My Image](https://raw.githubusercontent.com/ralf-42/Image/main/genai-banner-2.jpg)

---
Stand: 08.2025

# 1 Basismodule (1-12)

Die Basismodule bilden das Fundament des Kurses und vermitteln grundlegende Konzepte und Werkzeuge der generativen KI:

- **Einführende Grundlagen**: Allgemeine Einführung in generative KI, Modellsteuerung und fundamentale Frameworks (Module 1-4)
- **Technische Grundlagen**: Vertiefung in Transformer-Architektur, Memory-Konzepte und Output-Parser (Module 5-7)
- **Praktische Anwendungen**: Einführung in RAG, multimodale Bildverarbeitung, Agenten, Gradio und lokale Modelle (Module 8-12)

Diese Module stellen sicher, dass Sie über ein solides Grundverständnis der generativen KI-Technologien verfügen.

# 2 Erweiterungsmodule (13-23)

Die Erweiterungsmodule bauen auf den Grundlagen auf und bieten fortgeschrittene Konzepte und Spezialisierungen:

- **Erweiterte multimodale Anwendungen**: SQL RAG, Multimodal RAG, Audio- und Videoverarbeitung (Module 13-16)
- **Modelloptimierung**: MCP, Fine-Tuning, Modellauswahl und Evaluation (Module 17-19)
- **Fortgeschrittene Methoden**: Advanced Prompt Engineering und Context Engineering (Modul 20-21)
- **Regulatorische Aspekte**: EU AI Act und Ethik (Modul 22)
- **KI-Challenge**: Praktische Anwendung und Integration der Kursmodule (Modul 23)

Diese Module vertiefen spezifische Anwendungsbereiche und bieten fortgeschrittene Techniken für professionelle KI-Anwendungen.

**Die Basismodule sind obligatorisch, die Erweiterungsmodule sind fakultativ.**


# 3 Übersicht nach Basis/Erweiterung

| Modul-Nr. | Modultyp    | NB  | PDF | Inhalt                                | Themen                                                                                                                                                                                                                                          |                                                                Relevanz                                                                 |
| --------- | ----------- |:---:|:---:| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |:---------------------------------------------------------------------------------------------------------------------------------------:|
| 1         | Basis       |  ✓  |  ✓  | Einführung GenAI                      | • Kursüberblick<br>• Überblick Generative AI<br>• Einführung OpenAI<br>• Einführung Hugging Face<br>• Einführung LangChain                                                                                                                      |                            Fundamentales Verständnis der Gen-AI-Technologien <br>als Grundlage für den Kurs.                            |
| 2         | Basis       |  ✓  |     | Modellsteuerung -und optimierung      | • Überblick Prompting, Context <br>   Engineering,  RAG, Fine-Tuning<br>• Einsatzszenarien<br>• Entscheidungskriterien<br>• Trade-offs                                                                                                          |                              Essentielles Verständnis der verschiedenen Ansätze <br>zur Modellansteuerung.                              |
| 3         | Basis       |  ✓  |  ✓  | Codieren mit GenAI                    | • Prompting für Codegenerierung<br>• Revisionsprompts<br>• Debugging mit LLMs<br>• Prozessintegration<br>• Grenzen der LLM-Codegenerierung                                                                                                      |                                     Praktische Fähigkeiten für LLM-gestützte Programm-Entwicklung.                                      |
| 4         | Basis       |  ✓  |  ✓  | LangChain 101                         | • Was ist LangChain & Architektur<br>• Kernkonzepte (Chains, Models, Prompts)<br>• Best Practices & Design Patterns                                                                                                                             |              Fundamentales Verständnis von LangChain als zentrales <br>Framework für die Entwicklung von LLM-Anwendungen.               |
| 5         | Basis       |  ✓  |  ✓  | Large Language Models und Transformer | • Foundation Model<br>• Transformer-Architektur<br>• Textgenerierung<br>• Textzusammenfassung<br>• Textklassifizierung<br>• LLM schreibt ein Buch                                                                                               |                             Vertieftes Verständnis der Transformer-Architektur <br>und LLM-Funktionsweise.                              |
| 6         | Basis       |  ✓  |     | Chat und Memory                       | • Übersicht<br>• Kurzeit-Memory<br>• Langzeit-Memory<br>• Externes Memory                                                                                                                                                                       |                     Memory-Typen für Flexibilität und Skalierbarkeit, <br>um Konversationen effizient zu verwalten.                     |
| 7         | Basis       |  ✓  |     | Output Parser                         | • Structured Output Parser<br>• JSON<br>• DatetimeOutputParser<br>• Custom Output Parser                                                                                                                                                        |                               Parser-Typen für robuste und flexible Verarbeitung <br>von Modellantworten.                               |
| 8         | Basis       |  ✓  |  ✓  | Retrieval Augmented Generation        | • Einführung in RAG<br>• ChromaDB<br>• Embeddings<br>• Q&A über Dokumente<br>• Embedding-Datenbanken                                                                                                                                            |                            Schlüsseltechnologie für Informationsverarbeitung <br>mit unstrukturierten Daten.                            |
| 9         | Basis       |  ✓  |  ✓  | Multimodal Bild                       | • Bildgenerierung<br>• In-/Outpainting<br>• Bildklassifizierung<br>• Objekterkennung<br>• Bildbeschreibung                                                                                                                                      |                                                  Erweiterung um visuelle Komponenten.                                                   |
| 10        | Basis       |  ✓  |     | Agents                                | • Grundlagen von KI-Agenten<br>• Agentenarchitekturen<br>• Planung und Zielverfolgung<br>• Multi-Agenten-systeme                                                                                                                                |                                                    Entwicklung autonomer KI-Systeme.                                                    |
| 11        | Basis       |  ✓  |     | Gradio                                | • Grundkonzepte von Gradio<br>• Installation und Setup<br>• Zentrale Komponenten<br>• Praktische Beispiele<br>• Fortgeschrittene Konzepte<br>• Integration mit KI-Modellen<br>• Best Practices<br>• Deployment und Sharing                      |                                                   UI-Entwicklung für KI-Anwendungen.                                                    |
| 12        | Basis       |  ✓  |     | Lokale und Open Source Modelle        | • Einführung<br>• Ollama<br>• Lokale Modelle und LangChain<br>• Open Source vs. Closed Source<br>• Beispiele Open Source Modelle<br>• Lizenzierung und rechtliche Aspekte<br>• Auswahlkriterien Open Source<br>• Zukunftstrends bei Open Source |                                            Einsatz von KI-Modellen ohne Cloud-Abhängigkeit.                                             |
| 13        | Erweiterung |  ✓  |     | SQL RAG                               | • Einführung in SQL RAG<br>• Vergleich zu RAG<br>• Integration von LLMs mit Datenbanken<br>• SQL-Generierung mit LLMs<br>• Datenvalidierung und Sicherheitsaspekte<br>• Praktische Anwendungsfälle<br>• SQL RAG mit LangChain                   |                            Schlüsseltechnologie für die Arbeit mit <br>strukturierten Daten und Datenbanken.                            |
| 14        | Erweiterung |  ✓  |     | Multimodal RAG                        | • Intro<br>• Enhanced Document Processor <br>• Multimodal Embeddings<br>• Cross-Modale Suche                                                                                                                                                    |                                     Schlüsseltechnologie für die Arbeit mit <br>Texten und Bildern.                                     |
| 15        | Erweiterung |  ✓  |     | Multimodal Video                      | • Video-zu-Text (VTT)<br>• Text-zu-Video (TTV)<br>• Bild-zu-Video (ITV)<br>• Videoanalyse<br>• Video-Objekterkennung                                                                                                                            |                           Integration von Videoverarbeitung und -analyse<br> für multimodale KI-Anwendungen.                            |
| 16        | Erweiterung |  ✓  |  ✓  | Multimodal Audio                      | • Speech-to-Text (STT)<br>• Text-to-Speech (TTS)<br>• Sprachanalyse<br>• Audio-Summary<br>• Audio-Pipeline<br>• Podcast                                                                                                                         |                                                   Erschließung von Sprachanwendungen.                                                   |
| 17        | Erweiterung |  ✓  |     | MCP - Modell Context Protocol         | • Intro<br>• MCP-Server erstellen<br>• MCP-Client erstellen<br>• Al-Assistant mit MCP<br>• MCP in der Praxis                                                                                                                                    |                 Optimierung und effiziente Anpassung von <br> LLM Tooleinsatz durch ein standardisiertes Protokoll<br>                  |
| 18        | Erweiterung |  ✓  |  ✓  | Fine-Tuning                           | • Fine-Tuning mit Dashboard<br>• Fine-Tuning mit Code<br>• Parameter Efficient Fine-Tuning<br>• Bewerten des FT-Modells                                                                                                                         |                                Optimierung und effiziente Anpassung für <br>spezifische Anwendungsfälle.                                |
| 19        | Erweiterung |     |  ✓  | Modellauswahl und Evaluation          | • KI-Modelle<br>• Bewertung von Modellen<br>• Modellauswahlprozess<br>• Auswahlkriterien<br>• Benchmarks und Modellvergleiche                                                                                                                   |                                                 Fundierte Modellauswahl und Bewertung.                                                  |
| 20        | Erweiterung |     |  ✓  | Prompt Engineering                    | • Einführung Prompt-Engineering<br>• Few-Shot und Chain-of-Thought<br>• Persona- und Rollenmuster<br>• Frage- und Prüfmuster<br>• Inhalt- und Struktur-Muster<br>• Chat- und Reasoning-Modelle                                                  |                                                   Fortgeschrittene Prompt-Strategien.                                                   |
| 21        | Erweiterung |     |  ✓  | Context Engineering                   | • Grundstrategien<br>• Häufige Fehler<br>• Tools und Techniken<br>• Praktische Anwendung                                                                                                                                                        |                                                 Strategien für das Context Management.                                                  |
| 22        | Erweiterung |     |  ✓  | EU AI Act /Ethik                      | • Risikobasierte Einstufung    <br>• Hochrisiko-Anwendungen<br>• Strenge Auflagen für Hochrisiko-KI <br>• Transparenz- und Informationspflichten<br>• Ethische Grundsätze                                                                       | Der EU AI Act ist seit 2025 das zentrale Regelwerk für den<br> vertrauenswürdigen, sicheren und ethischen <br>Einsatz von KI in Europa. |
| 23        | Erweiterung |     |  ✓  | KI-Challenge                          | • Projektoptionen <br>• Integration mehrerer Technologien<br>• Entwicklung einer E2E-Anwendung<br>• Gradio-Benutzeroberfläche<br>• Evaluation und Dokumentation                                                                                 |                               Anwendung und Integration der<br>erlernten Konzepte in einer Gesamtlösung.                                |

<div style="page-break-after: always;"></div>


# 4 Kursbibliothek *genai_lib*

Diese Kursbibliothek stellt eine Sammlung nützlicher Hilfsfunktionen für KI- und Machine Learning-Projekte bereit, insbesondere für die Arbeit mit LangChain und Large Language Models in Jupyter-Notebooks.

🛠️ utilities.py

**Übersicht**

Eine Sammlung von Standard-Hilfsfunktionen für KI- und LangChain-Projekte in Python-Entwicklungsumgebungen, insbesondere für Google Colab.


 `check_environment()`

- Zeigt die aktuelle Python-Version an
- Listet alle installierten LangChain-Bibliotheken auf
- Unterdrückt störende Deprecation-Warnungen

`install_packages(modules)`

- Installiert Python-Module automatisch mit `uv pip install`
- Prüft vorher, ob Module bereits verfügbar sind
- Optimiert für Google Colab mit ruhiger Installation

 `get_ipinfo()`

- Ruft Geoinformationen zur aktuellen öffentlichen IP-Adresse ab
- Zeigt IP, Standort, Provider und weitere Netzwerkdaten an

 `setup_api_keys(key_names, create_globals=True)`

- Lädt API-Keys aus Google Colab userdata
- Setzt sie als Umgebungsvariablen und optional als globale Variablen
- Unterstützt mehrere Keys gleichzeitig (OpenAI, Anthropic, Hugging Face, etc.)

`mprint(text)`

- Gibt Text als formatiertes Markdown in Jupyter-Notebooks aus
- Nutzt IPython's `display()` und `Markdown()` für bessere Darstellung

`process_response(response)`

- Extrahiert strukturierte Informationen aus LLM-Antworten
- Parst Token-Nutzungsdaten (Prompt, Completion, Total)
- Gibt bereinigten Text und Metadaten als Dictionary zurück



## 4.1 Verwendung

```python
from utilities import check_environment, setup_api_keys, mprint, install_packages
# Umgebung prüfen
check_environment()

# API-Keys setzen
setup_api_keys(["OPENAI_API_KEY", "ANTHROPIC_API_KEY"])

# Markdown ausgeben
mprint("# Hallo **Welt**!")

# Installation fehlender Module
install_packages(['langchain', 'langchain_openai', 'langchain_community', 'openai', 'gradio'])
```

## 4.2 Zielgruppe

Diese Utility-Bibliothek ist speziell für KI-Entwickler und Data Scientists konzipiert, die in Google Colab oder ähnlichen Jupyter-Umgebungen arbeiten und häufig mit LangChain und verschiedenen AI-APIs interagieren.


📋 prepare_prompt.py


`apply_prepare_framework(task, role, tone, word_limit)`
- **Zweck**: Erstellt strukturierte Prompts nach dem PREPARE-Framework
- **Parameter**: Aufgabe, Rolle, Tonfall, Wortlimit
- **Ausgabe**: Vollständig formatierter Prompt mit 7 Komponenten (P-R-E-P-A-R-E)
- **Anwendung**: Systematische Prompt-Erstellung für bessere LLM-Ergebnisse


🎨 show_md.py

Markdown-Display-Funktionen für Jupyter Notebooks:

- **`show_md(text, prefix)`** - Basis-Markdown-Anzeige mit optionalem Prefix
- **`show_title(text)`** - Zeigt Titel mit 💡-Emoji (`# Titel 💡`)
- **`show_subtitle(text)`** - Zeigt Untertitel (`## Untertitel`)
- **`show_info(text)`** - Info-Meldung mit ℹ️-Symbol
- **`show_warning(text)`** - Warnung mit ⚠️-Symbol
- **`show_success(text)`** - Erfolgsmeldung mit ✅-Symbol

**Anwendung**: Strukturierte und visuell ansprechende Notebook-Ausgaben


🎯 Typische Anwendung

```python
# 1. Umgebung prüfen und API-Keys laden
check_environment()
setup_api_keys(["OPENAI_API_KEY"])

# 2. Strukturierte Notebook-Ausgaben
show_title("Mein KI-Projekt")
show_info("Experiment gestartet")

# 3. PREPARE-Prompts erstellen
prompt = apply_prepare_framework(
    task="Erkläre maschinelles Lernen",
    role="KI-Tutor",
    tone="verständlich"
)

# 4. LLM-Antworten verarbeiten
result = process_response(llm_response)
show_success(f"Antwort erhalten: {result['tokens_total']} Tokens")
```

**Ideal für**: KI-Kurse, Jupyter-Notebooks, LangChain-Projekte, strukturierte Prompt-Engineering

📌 Umgebung einrichten 

```Python
!uv pip install --system --prerelease allow -q git+https://github.com/ralf-42/Python_Modules
from genai_lib.utilities import check_environment, get_ipinfo, setup_api_keys, mprint
setup_api_keys(['OPENAI_API_KEY', 'HF_TOKEN'], create_globals=False)
print()
check_environment()
print()
get_ipinfo()
```
