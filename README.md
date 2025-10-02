# 🤖 Kursbibliothek Generative KI und ML

Diese Bibliothek stellt Hilfsmittel und Funktionen für den Kurs **"Generative KI"** bereit. Sie erleichtert Teilnehmer:innen den Einstieg und die praktische Anwendung generativer KI-Technologien in Google Colab-Umgebungen.

## 1. 📦 Installation

Die Bibliothek ist für die Installation in Google Colab-Umgebungen optimiert:

```bash
!uv pip install -q git+https://github.com/ralf-42/Python_modules
```

Nach der Installation können die Module importiert werden:

```python
from genai_lib.utilities import setup_api_keys, install_packages
from genai_lib.llm_basics import setup_ChatOpenAI
from genai_lib.show_md import show_title, show_info
```

## 2. 🏗️ Architektur und Module

Die Bibliothek besteht aus modularen Hilfsdateien im `genai_lib/` Verzeichnis:

- **`utilities.py`** - Kernfunktionen für Umgebungssetup, API-Schlüssel-Verwaltung, Paketinstallation und Antwortverarbeitung
- **`llm_basics.py`** - Grundlegende LLM-Setup-Funktionen, hauptsächlich für ChatOpenAI-Konfiguration
- **`chromadb_statistics.py`** - Umfassende ChromaDB-Analyse und Statistik-Modul mit detaillierter Collection-Analytik
- **`prepare_prompt.py`** - PREPARE-Framework-Implementierung für strukturierte Prompt-Entwicklung
- **`show_md.py`** - Jupyter Notebook Display-Utilities für formatierte Markdown-Ausgabe

## 3. 📋 Abhängigkeiten

Die Bibliothek benötigt folgende LangChain- und KI-bezogene Pakete:

- `langchain_openai`
- `langchain-community` 
- `langchain-text-splitters`
- `langchain_experimental`
- `langchain-ollama`
- `chromadb`

Eine vollständige Liste der Abhängigkeiten finden Sie in der `requirements.txt`.

## 4. 🚀 Wichtige Funktionen

### API-Schlüssel-Verwaltung
- `setup_api_keys()` - Richtet API-Schlüssel aus Google Colab userdata für OpenAI, Anthropic, Hugging Face ein
- `install_packages()` - Installiert automatisch benötigte Pakete, falls nicht verfügbar

### ChromaDB-Analyse
Das `chromadb_statistics.py` Modul bietet umfassende Datenbankanalyse:
- `display_chromadb_statistics()` - Vollständige Datenbankstatistik-Übersicht
- `analyze_collection()` - Detaillierte Analyse einzelner Collections
- `get_collection_chunks()` - Abrufen und Untersuchen einzelner Chunks
- Interaktives CLI-Menü mit über 10 Analyseoptionen

### LLM-Utilities
- `setup_ChatOpenAI()` - Schnelles ChatOpenAI-Setup mit Standardparametern (gpt-4o-mini, temp=0.0)
- `process_response()` - Strukturierte Informationen aus LLM-Antworten extrahieren inkl. Token-Verwendung

### Display-Utilities
- `mprint()` / `show_md()` - Markdown-Anzeige in Notebooks
- `show_title()`, `show_info()`, `show_warning()`, `show_success()` - Formatierte Benachrichtigungsfunktionen

## 5. 💡 Verwendung

Typische Verwendung in Google Colab-Lernumgebungen:

1. Installation über uv pip in Colab
2. Import der Utilities: `from genai_lib.utilities import setup_api_keys, install_packages`
3. Umgebungssetup mit `setup_api_keys(["OPENAI_API_KEY", "ANTHROPIC_API_KEY"])`
4. Verwendung der Module für Kursübungen mit LangChain und ChromaDB

## 6. 🛠️ Entwicklung

### Build und Installation
```bash
# Installation im Entwicklungsmodus
pip install -e .

# Paket erstellen
python setup.py sdist bdist_wheel
```

### Modulausführung
Einzelne Module können direkt ausgeführt werden:
```bash
python genai_lib/chromadb_statistics.py  # Interaktives ChromaDB-Analyse-Tool
python genai_lib/llm_basics.py          # LLM-Modellattribute anzeigen
```

## 7. 📁 Dateiorganisation

```
Python_modules/
├── genai_lib/
│   ├── __init__.py          # Leerer Modul-Initializer
│   ├── utilities.py         # Kern-Utilities
│   ├── llm_basics.py        # LLM-Setup-Funktionen
│   ├── chromadb_statistics.py # ChromaDB-Analyse-Tool
│   ├── prepare_prompt.py    # PREPARE-Framework
│   └── show_md.py          # Display-Utilities
├── ml_lib/                  # Zusätzliches Modul (nicht Teil der Installation)
│   ├── __init__.py
│   └── utilities.py         # ML-spezifische Utilities
├── README.md
├── requirements.txt
└── setup.py
```

### Ignorierte Dateien und Verzeichnisse

Die folgenden Dateien und Verzeichnisse werden durch `.gitignore` und `.claudeignore` von der Versionskontrolle ausgeschlossen:

**Python-spezifisch:**
- `__pycache__/`, `*.pyc` - Python-Bytecode
- `*.egg-info/`, `.pytest_cache/` - Build- und Test-Artefakte
- `.ipynb_checkpoints/` - Jupyter Notebook Checkpoints
- `.venv/`, `venv/` - Virtuelle Umgebungen

**Projektspezifisch:**
- `_misc/` - Verschiedene Hilfsdateien
- `.tmp.drivedownload`, `.tmp.driveupload` - Temporäre Drive-Dateien
- `*.pptx`, `*.png`, `*.jpeg` - Präsentationen und Bilder
- `.obsidian/` - Obsidian-Notizen
- `CLAUDE.md` - Claude Code Projektanweisungen

## 8. 📄 Lizenz

Dieses Projekt steht unter der **MIT-Lizenz**. Die Kursmaterialien können frei verwendet, modifiziert und weiterverbreitet werden.

**MIT License - Copyright (c) 2025 Ralf**

Weitere Details finden Sie in der `LICENSE`-Datei.

