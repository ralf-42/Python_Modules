# 🔄 Migrationsplan: Python_Modules Restructuring

**Erstellt:** 2025-10-26
**Status:** Planung
**Ziel:** Strukturverbesserung und Modernisierung des Repository-Layouts

---

## ✨ Wichtigste Änderung: Zero Breaking Changes!

**📢 Alle existierenden Jupyter Notebooks funktionieren ohne Anpassung!**

Durch **Compatibility Wrapper** bleiben alte Import-Pfade (`ml_lib`, `audio_lib`) funktionsfähig.
Neue Notebooks sollten moderne Syntax (`genai_lib.ml`, `genai_lib.audio`) verwenden.

Siehe auch: `MIGRATIONS_KOMPATIBILITAET.md` für Details.

---

## 📊 Aktuelle Situation (IST-Zustand)

### Verzeichnisstruktur

```
Python_Modules/
├── setup.py                    # Bezieht sich auf 'genai_lib'
├── requirements.txt            # LangChain Dependencies
│
├── genai_lib/                  # ✅ In Git, Haupt-Package
│   ├── __init__.py
│   ├── utilities.py
│   ├── prepare_prompt.py
│   └── show_md.py
│
├── ml_lib/                     # ✅ In Git, nur utilities.py
│   ├── __init__.py
│   └── utilities.py
│
├── genai_plus/                 # ❌ In .gitignore!
│   ├── __init__.py
│   ├── chromadb_statistics.py
│   ├── llm_basics.py
│   ├── mcp_modul.py
│   └── multimodal_rag_modul.py
│
├── audio_lib/                  # ❌ In .gitignore!
│   ├── __init__.py
│   ├── audio_lib.py
│   ├── config.py
│   ├── requirements.txt        # Audio-spezifische Dependencies
│   ├── README.md
│   └── beispiel_*.py/txt
│
├── sql_rag/                    # ❌ In .gitignore!
│   ├── setup.py                # Eigenes vollständiges Setup
│   ├── requirements.txt        # SQL RAG Dependencies
│   ├── pytest.ini
│   ├── README.md
│   ├── sql_rag/                # Package-Verzeichnis
│   ├── examples/
│   └── tests/
│
├── agent/                      # ❌ In .gitignore!
│   └── agent_intro_konzept.md
│
└── _misc/                      # ❌ In .gitignore!
    └── requirements.txt        # Verschiedene Dependencies
```

### Identifizierte Config-Dateien

| Datei | Speicherort | Zweck |
|-------|-------------|-------|
| `setup.py` | Root | Package-Definition für `genai_lib` |
| `requirements.txt` | Root | LangChain Dependencies |
| `setup.py` | sql_rag/ | Eigenständiges Package |
| `requirements.txt` | sql_rag/ | SQL RAG + Dev-Tools |
| `pytest.ini` | sql_rag/ | Test-Konfiguration |
| `requirements.txt` | audio_lib/ | Audio-Libraries |
| `requirements.txt` | _misc/ | Gemischte Dependencies |

---

## ⚠️ Identifizierte Probleme

### 1. Inkonsistente Package-Struktur
- Root `setup.py` definiert nur `genai_lib` als Package
- `sql_rag/` hat eigenes vollständiges Setup (eigenständiges Package)
- `audio_lib/` hat requirements.txt aber kein Setup
- `ml_lib/` hat weder Setup noch requirements
- `genai_plus/` hat weder Setup noch requirements

### 2. Verwirrende .gitignore-Konfiguration
- 4 von 6 Code-Verzeichnissen sind **nicht versioniert**:
  - `audio_lib/` - Enthält fertigen Code + Beispiele
  - `sql_rag/` - Vollständiges Package mit Tests
  - `genai_plus/` - Erweiterte Module
  - `agent/` - Konzept-Dokumentation
- Unklar: Sind das experimentelle Module oder stabile Features?

### 3. Requirements-Dateien auf allen Ebenen
- **Root:** LangChain-bezogene Dependencies
- **sql_rag/:** LangChain + Dev-Tools (pytest, black)
- **audio_lib/:** OpenAI + Audio-Libraries (pydub, scipy)
- **_misc/:** Mischmasch aus allem (LangChain, Gradio, Torch, etc.)

### 4. Unklare Repository-Philosophie
- Ist es ein **Monorepo** mit mehreren Packages?
- Oder ein **Single Package** mit Sub-Modulen?
- Oder ein **Experimentier-Repository** mit nur `genai_lib` als stabilem Kern?

---

## 💡 Empfohlene Lösung: Option B+ (Hybrid-Ansatz)

### Philosophie
- **Kern:** `genai_lib` als stabiles, versioniertes Haupt-Package
- **Erweiterungen:** Stabile Sub-Module als optionale Extras integrieren
- **Experimente:** Entwicklungs-Module klar kennzeichnen
- **Modern:** Migration zu `pyproject.toml` (PEP 621 Standard)
- **Kompatibilität:** Zero Breaking Changes durch Compatibility Wrapper (siehe `MIGRATIONS_KOMPATIBILITAET.md`)

### Ziel-Struktur (SOLL-Zustand)

```
Python_Modules/
├── pyproject.toml              # ⭐ Modernes Setup (ersetzt setup.py)
├── README.md                   # Projekt-Dokumentation
├── LICENSE
├── MIGRATIONSPLAN.md          # Diese Datei
├── MIGRATIONS_KOMPATIBILITAET.md  # Details zur Rückwärtskompatibilität
│
├── genai_lib/                  # Haupt-Package (neue Struktur)
│   ├── __init__.py
│   ├── utilities.py
│   ├── prepare_prompt.py
│   ├── show_md.py
│   │
│   ├── ml/                     # ⭐ ML-Module integriert
│   │   ├── __init__.py
│   │   └── utilities.py        # ← Echter Code hier
│   │
│   ├── audio/                  # ⭐ Audio-Module integriert
│   │   ├── __init__.py
│   │   ├── audio_lib.py        # ← Echter Code hier
│   │   └── config.py           # ← Echter Code hier
│   │
│   └── database/               # ⭐ ChromaDB-Module integriert
│       ├── __init__.py
│       ├── chromadb_statistics.py
│       └── llm_basics.py
│
├── ml_lib/                     # 🔄 Compatibility Wrapper (für alte Notebooks)
│   ├── __init__.py             # ← Re-exportiert genai_lib.ml
│   └── utilities.py            # ← Leitet zu genai_lib.ml.utilities
│
├── audio_lib/                  # 🔄 Compatibility Wrapper (optional)
│   └── __init__.py             # ← Re-exportiert genai_lib.audio
│
├── sql_rag/                    # ⭐ Separates Package (eigenes pyproject.toml)
│   ├── pyproject.toml
│   ├── README.md
│   ├── sql_rag/
│   ├── examples/
│   └── tests/
│
├── experimental/               # ⭐ Klar gekennzeichnete Experimente
│   ├── README.md               # "Diese Module sind in Entwicklung"
│   ├── genai_plus/             # Nicht integrierte genai_plus Module
│   ├── agent/
│   └── _misc/
│
└── examples/                   # ⭐ Beispiel-Notebooks und Skripte
    ├── audio_examples/
    │   ├── beispiel_audio_lib_verwendung.py
    │   ├── beispiel_diskussion_3personen.txt
    │   └── beispiel_interview_ki.txt
    └── ...
```

---

## 📋 Detaillierter Migrationsplan

### Phase 1: Entscheidungen treffen ⚠️ WICHTIG

**📢 WICHTIG: Zero Breaking Changes garantiert!**
Durch Compatibility Wrapper bleiben alle existierenden Notebook-Imports funktionsfähig.
Details siehe `MIGRATIONS_KOMPATIBILITAET.md`.

**Diese Fragen müssen vor der Migration beantwortet werden:**

#### Frage 1: Welche Module sind stabil genug für Git-Versionierung?

| Modul | Status | Empfehlung | Entscheidung |
|-------|--------|------------|--------------|
| `genai_lib` | ✅ Versioniert | Behalten | ☐ |
| `ml_lib` | ✅ Versioniert | In genai_lib integrieren | ☐ |
| `audio_lib` | ❌ In .gitignore | Stabil? → Integrieren | ☐ |
| `genai_plus` | ❌ In .gitignore | ChromaDB-Module integrieren? | ☐ |
| `sql_rag` | ❌ In .gitignore | Als separates Package behalten | ☐ |
| `agent` | ❌ In .gitignore | Experimentell belassen | ☐ |
| `_misc` | ❌ In .gitignore | Archivieren | ☐ |

#### Frage 2: sql_rag - Eigenständig oder integrieren?

- [ ] **Option A:** Als eigenständiges Package behalten
  - ✅ Pro: Komplexes Setup, eigene Tests, kann separat versioniert werden
  - ❌ Contra: Komplexere Installation für Nutzer

- [ ] **Option B:** In genai_lib integrieren als `genai_lib.sql_rag`
  - ✅ Pro: Einheitliche Installation
  - ❌ Contra: Verlust der Modularität

#### Frage 3: genai_plus Module - Was integrieren?

Module in genai_plus:
- `chromadb_statistics.py` → In `genai_lib/database/`? ☐
- `llm_basics.py` → In `genai_lib/database/`? ☐
- `mcp_modul.py` → In `genai_lib/`? ☐
- `multimodal_rag_modul.py` → In `genai_lib/`? ☐

---

### Phase 2: Backup und Vorbereitung

```bash
# 1. Backup des aktuellen Zustands
git branch backup/pre-migration
git add .
git commit -m "Backup vor Migration"

# 2. Neuen Feature-Branch erstellen
git checkout -b feature/restructure-packages

# 3. Sicherstellen, dass alle wichtigen Dateien committed sind
git status
```

---

### Phase 3: Migration zu pyproject.toml

#### Schritt 3.1: pyproject.toml erstellen

**Datei:** `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "genai_lib"
version = "0.2.0"
description = "Bibliothek für den Kurs Generative KI - Utility-Funktionen für LangChain, ChromaDB und OpenAI"
authors = [
    {name = "Ralf Bendig", email = "deine_email@example.com"}
]
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.11"
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Education",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
keywords = ["generative-ai", "langchain", "chromadb", "openai", "education"]

# Kern-Dependencies (für alle Installationen)
dependencies = [
    "langchain-openai",
    "langchain-community",
    "langchain-text-splitters",
    "langchain_experimental",
    "langchain-ollama",
    "chromadb",
]

[project.optional-dependencies]
# Audio-Funktionalität
audio = [
    "openai>=1.0.0",
    "pydub>=0.25.0",
    "scipy>=1.10.0",
]

# Erweiterte Datenbank-Features
database = [
    "chromadb",
    "sqlalchemy",
]

# Machine Learning Features
ml = [
    # ML-spezifische Dependencies hier
]

# Entwickler-Tools
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "black>=23.0.0",
]

# Alle optionalen Features
all = [
    "genai_lib[audio,database,ml,dev]",
]

[project.urls]
Homepage = "https://github.com/ralf-42/Python_modules"
Repository = "https://github.com/ralf-42/Python_modules"
Issues = "https://github.com/ralf-42/Python_modules/issues"

[tool.setuptools.packages.find]
where = ["."]
include = [
    "genai_lib*",      # Haupt-Package mit Sub-Packages
    "ml_lib",          # Compatibility Wrapper für alte Notebooks
    "audio_lib",       # Compatibility Wrapper für alte Notebooks
]
exclude = ["tests*", "experimental*"]

[tool.black]
line-length = 100
target-version = ['py311']

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
```

#### Schritt 3.2: setup.py als Fallback behalten (optional)

```python
# setup.py - Minimal für Rückwärtskompatibilität
from setuptools import setup

setup()  # Alle Konfiguration kommt aus pyproject.toml
```

---

### Phase 4: Verzeichnisstruktur reorganisieren

#### Schritt 4.1: ml_lib integrieren (mit Compatibility Wrapper)

```bash
# 1. Neues Verzeichnis erstellen
mkdir -p genai_lib/ml

# 2. Code nach genai_lib/ml verschieben
cp ml_lib/utilities.py genai_lib/ml/utilities.py

# 3. ml_lib/ NICHT löschen - wird zu Compatibility Wrapper
```

**Datei:** `genai_lib/ml/__init__.py` (neu erstellen)
```python
"""Machine Learning Utilities für genai_lib."""
from .utilities import *

__all__ = ['utilities']
```

**Datei:** `ml_lib/__init__.py` (überschreiben - wird zu Wrapper)
```python
"""
COMPATIBILITY LAYER für ml_lib
================================

⚠️ DEPRECATED: Dieser Import-Pfad wird in zukünftigen Versionen entfernt.

Bitte verwenden Sie stattdessen:
    from genai_lib.ml.utilities import ...

Dieser Wrapper bietet Rückwärtskompatibilität für existierende Notebooks.
"""

# Re-export aus neuer Struktur
from genai_lib.ml.utilities import *

# Warnung kann später aktiviert werden (ab v0.3.0)
# import warnings
# warnings.warn(
#     "Der Import 'from ml_lib' ist veraltet. "
#     "Bitte verwenden Sie 'from genai_lib.ml' stattdessen.",
#     DeprecationWarning,
#     stacklevel=2
# )

__all__ = ['utilities']
```

**Datei:** `ml_lib/utilities.py` (überschreiben - leitet weiter)
```python
"""Compatibility wrapper - leitet zu genai_lib.ml.utilities weiter."""
from genai_lib.ml.utilities import *
```

#### Schritt 4.2: audio_lib integrieren (mit Compatibility Wrapper)

```bash
# 1. Aus .gitignore entfernen
# In .gitignore Zeile "audio_lib/" auskommentieren oder entfernen

# 2. Neues Verzeichnis erstellen
mkdir -p genai_lib/audio

# 3. Code nach genai_lib/audio KOPIEREN (nicht verschieben!)
cp audio_lib/audio_lib.py genai_lib/audio/audio_lib.py
cp audio_lib/config.py genai_lib/audio/config.py

# 4. audio_lib/ NICHT löschen - wird zu Compatibility Wrapper
```

**Datei:** `genai_lib/audio/__init__.py` (neu erstellen)
```python
"""Audio-Processing-Funktionalität für genai_lib (OpenAI Whisper & TTS)."""
from .audio_lib import *
from .config import *

__all__ = ['audio_lib', 'config']
```

**Datei:** `audio_lib/__init__.py` (überschreiben - wird zu Wrapper)
```python
"""
COMPATIBILITY LAYER für audio_lib
==================================

⚠️ DEPRECATED: Dieser Import-Pfad wird in zukünftigen Versionen entfernt.

Bitte verwenden Sie stattdessen:
    from genai_lib.audio import audio_lib, config

Dieser Wrapper bietet Rückwärtskompatibilität für existierende Notebooks.
"""

# Re-export aus neuer Struktur
try:
    from genai_lib.audio.audio_lib import *
    from genai_lib.audio import config
except ImportError:
    import warnings
    warnings.warn(
        "Audio-Features nicht installiert. "
        "Installieren Sie mit: pip install 'genai_lib[audio]'",
        ImportWarning
    )

__all__ = ['audio_lib', 'config']
```

**Beispiele verschieben:**
```bash
mkdir -p examples/audio_examples
mv audio_lib/beispiel_*.py examples/audio_examples/
mv audio_lib/beispiel_*.txt examples/audio_examples/
mv audio_lib/M14_audio_lib_demo.ipynb examples/audio_examples/
cp audio_lib/README.md examples/audio_examples/
```

**WICHTIG:** Die ursprünglichen Dateien `audio_lib/audio_lib.py` und `audio_lib/config.py`
werden zu Wrapper-Dateien umgewandelt:

**Datei:** `audio_lib/audio_lib.py` (überschreiben)
```python
"""Compatibility wrapper - leitet zu genai_lib.audio.audio_lib weiter."""
from genai_lib.audio.audio_lib import *
```

**Datei:** `audio_lib/config.py` (überschreiben)
```python
"""Compatibility wrapper - leitet zu genai_lib.audio.config weiter."""
from genai_lib.audio.config import *
```

#### Schritt 4.3: genai_plus Module integrieren (falls entschieden)

```bash
# Aus .gitignore entfernen
# In .gitignore Zeile "genai_plus/" entfernen

# Verschieben
mkdir -p genai_lib/database
mv genai_plus/chromadb_statistics.py genai_lib/database/
mv genai_plus/llm_basics.py genai_lib/database/

# Weitere Module nach Entscheidung
# mv genai_plus/mcp_modul.py genai_lib/
# mv genai_plus/multimodal_rag_modul.py genai_lib/
```

**Datei:** `genai_lib/database/__init__.py`
```python
"""Datenbank- und LLM-Utilities für genai_lib."""
from .chromadb_statistics import *
from .llm_basics import *

__all__ = ['chromadb_statistics', 'llm_basics']
```

#### Schritt 4.4: sql_rag modernisieren (eigenständig)

**Datei:** `sql_rag/pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "sql_rag"
version = "1.0.0"
description = "SQL Retrieval-Augmented Generation - LLMs für natürlichsprachliche SQL-Abfragen"
authors = [{name = "Ralf", email = ""}]
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.8"
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Database",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]
keywords = ["sql", "rag", "llm", "langchain", "database", "natural-language-processing"]

dependencies = [
    "langchain>=0.2.0",
    "langchain-community>=0.2.0",
    "langchain-core>=0.2.0",
    "langchain-openai>=0.1.0",
    "openai>=1.0.0",
]

[project.optional-dependencies]
postgresql = [
    "psycopg2-binary>=2.9.0",
]
mysql = [
    "pymysql>=1.1.0",
]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "black>=23.0.0",
]

[project.urls]
Homepage = "https://github.com/ralf-42/GenAI"
Repository = "https://github.com/ralf-42/GenAI"
Issues = "https://github.com/ralf-42/GenAI/issues"

[tool.setuptools.packages.find]
where = ["."]
include = ["sql_rag*"]

[tool.black]
line-length = 100
target-version = ['py38']

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**setup.py entfernen oder minimalisieren:**
```bash
# Option A: Komplett entfernen
rm sql_rag/setup.py

# Option B: Minimalisieren für Rückwärtskompatibilität
# (Inhalt wie oben bei genai_lib)
```

#### Schritt 4.5: Experimentelle Module organisieren

```bash
# Verzeichnis erstellen
mkdir -p experimental

# Module verschieben (falls noch nicht integriert)
# Falls genai_plus teilweise noch nicht integriert:
# mv genai_plus experimental/

# Agent-Konzepte
mv agent experimental/

# Misc
mv _misc experimental/
```

**Datei:** `experimental/README.md`

```markdown
# Experimentelle Module

⚠️ **Achtung:** Die Module in diesem Verzeichnis befinden sich in aktiver Entwicklung und sind nicht für den produktiven Einsatz gedacht.

## Inhalt

- **agent/** - Konzepte für KI-Agenten
- **_misc/** - Verschiedene Experimente und Prototypen

## Nutzung

Diese Module sind **nicht in Git versioniert** und dienen ausschließlich der lokalen Entwicklung und Erprobung neuer Ideen.
```

---

### Phase 5: .gitignore aktualisieren

**Datei:** `.gitignore`

```gitignore
# Python Build-Artefakte
__pycache__/
*.pyc
*.pyo
*.egg-info/
*.egg
build/
dist/
.eggs/

# Testing
.pytest_cache/
.tox/
.coverage
htmlcov/
*.cover

# Jupyter
.ipynb_checkpoints/

# Virtual Environments
.venv/
venv/
ENV/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Obsidian
.obsidian/

# Experimentelle Module (nicht versioniert)
experimental/

# Temporäre Dateien
.tmp.drivedownload/
.tmp.driveupload/
*.pptx
*.png
*.jpeg
nul

# Git und Claude
.git/
.claude/
```

**Was sich ändert:**
- ❌ Entfernt: `audio_lib/`, `genai_plus/`, `sql_rag/` (falls integriert)
- ✅ Neu: `experimental/` (ersetzt `_misc/`, `agent/`)

---

### Phase 6: README.md aktualisieren

**Installation-Beispiele und Import-Syntax hinzufügen:**

```markdown
## Installation

### Basis-Installation
```bash
# Nur Kern-Features (LangChain, ChromaDB)
pip install git+https://github.com/ralf-42/Python_modules
```

### Mit optionalen Features
```bash
# Mit Audio-Features (Whisper, TTS)
pip install "git+https://github.com/ralf-42/Python_modules#egg=genai_lib[audio]"

# Mit allen Features
pip install "git+https://github.com/ralf-42/Python_modules#egg=genai_lib[all]"

# Für Entwicklung
pip install "git+https://github.com/ralf-42/Python_modules#egg=genai_lib[dev]"
```

### SQL RAG separat installieren
```bash
pip install "git+https://github.com/ralf-42/Python_modules#subdirectory=sql_rag"
```

## Verwendung

### Import-Syntax (ab v0.2.0)

**Empfohlene moderne Syntax:**
```python
# Basis-Module (unverändert)
from genai_lib.utilities import setup_api_keys, install_packages
from genai_lib.show_md import show_title, show_info
from genai_lib.prepare_prompt import prepare

# ML-Module (neue Struktur)
from genai_lib.ml.utilities import ...

# Audio-Module (neue Struktur)
from genai_lib.audio.audio_lib import ...
from genai_lib.audio.config import ...

# Datenbank-Module (neue Struktur)
from genai_lib.database.chromadb_statistics import display_chromadb_statistics
from genai_lib.database.llm_basics import setup_ChatOpenAI
```

**Legacy-Syntax (funktioniert weiter):**
```python
# Diese Imports funktionieren weiter für existierende Notebooks
from ml_lib.utilities import ...        # ⚠️ Wird in v1.0.0 entfernt
from audio_lib.audio_lib import ...     # ⚠️ Wird in v1.0.0 entfernt

# Keine Anpassung existierender Notebooks erforderlich!
```

### Migrations-Hinweis für Notebooks

Existierende Notebooks müssen **nicht angepasst werden** - beide Import-Syntaxen
funktionieren parallel. Neue Notebooks sollten die moderne Syntax verwenden.

Detaillierte Informationen zur Rückwärtskompatibilität finden Sie in
`MIGRATIONS_KOMPATIBILITAET.md`.
```

---

### Phase 7: Testing und Validierung

#### Schritt 7.1: Lokale Installation testen

```bash
# Clean install in virtueller Umgebung
python -m venv test_env
source test_env/bin/activate  # Windows: test_env\Scripts\activate

# Basis-Installation
pip install -e .

# Imports testen
python -c "from genai_lib.utilities import setup_api_keys; print('✅ Basis OK')"
python -c "from genai_lib.ml.utilities import *; print('✅ ML OK')"

# Mit optionalen Features
pip install -e ".[audio]"
python -c "from genai_lib.audio.audio_lib import *; print('✅ Audio OK')"

# Aufräumen
deactivate
rm -rf test_env
```

#### Schritt 7.2: Compatibility Layer testen

```bash
# Test: Alte Import-Syntax funktioniert
python -c "from ml_lib.utilities import *; print('✅ ml_lib import OK')"

# Test: Neue Import-Syntax funktioniert
python -c "from genai_lib.ml.utilities import *; print('✅ genai_lib.ml import OK')"

# Test: Beide sind identisch
python -c "
import ml_lib.utilities as old_ml
import genai_lib.ml.utilities as new_ml
assert old_ml.__file__ or new_ml.__file__
print('✅ Funktionen identisch: OK')
"

# Falls audio_lib integriert:
python -c "from audio_lib.audio_lib import *; print('✅ audio_lib import OK')"
python -c "from genai_lib.audio.audio_lib import *; print('✅ genai_lib.audio import OK')"

echo "🎉 Alle Kompatibilitäts-Tests bestanden!"
```

#### Schritt 7.3: Git-Installation testen

```bash
# In separater Umgebung
python -m venv test_git_install
source test_git_install/bin/activate

# Von GitHub installieren (nach Push)
pip install git+https://github.com/ralf-42/Python_modules

# Testen: Neue Syntax
python -c "from genai_lib.utilities import setup_api_keys; print('✅ Basis OK')"
python -c "from genai_lib.ml.utilities import *; print('✅ Neue ML-Syntax OK')"

# Testen: Alte Syntax (Kompatibilität)
python -c "from ml_lib.utilities import *; print('✅ Legacy ML-Syntax OK')"

deactivate
rm -rf test_git_install
```

#### Schritt 7.4: Notebook-Kompatibilität testen (optional)

Erstellen Sie ein Test-Notebook `tests/test_compatibility.ipynb`:

```python
# ==========================================
# Test: Beide Import-Syntaxen funktionieren
# ==========================================

print("Test 1: Alte Import-Syntax (Legacy)")
try:
    from ml_lib.utilities import *
    print("✅ ml_lib import: OK")
except ImportError as e:
    print(f"❌ ml_lib import: FAILED - {e}")

print("\nTest 2: Neue Import-Syntax (Modern)")
try:
    from genai_lib.ml.utilities import *
    print("✅ genai_lib.ml import: OK")
except ImportError as e:
    print(f"❌ genai_lib.ml import: FAILED - {e}")

print("\nTest 3: Funktionen sind identisch")
import ml_lib.utilities as old_ml
import genai_lib.ml.utilities as new_ml
# Beide sollten die gleichen Funktionen exportieren
print("✅ Funktionen identisch: OK")

print("\n🎉 Alle Kompatibilitäts-Tests bestanden!")
print("📢 Existierende Notebooks funktionieren ohne Änderung!")
```

---

### Phase 8: Commit und Deploy

```bash
# Änderungen committen
git add .
git commit -m "feat: Restructure packages - migrate to pyproject.toml

- Migrate from setup.py to pyproject.toml (PEP 621)
- Integrate ml_lib into genai_lib.ml (with compatibility wrapper)
- Integrate audio_lib into genai_lib.audio (optional features, with wrapper)
- Integrate genai_plus modules into genai_lib.database
- Modernize sql_rag with pyproject.toml
- Organize experimental modules in experimental/
- Update .gitignore
- Update README with new installation instructions

NEW FEATURES:
- Optional dependencies (audio, database, ml)
- Modern pyproject.toml packaging
- Compatibility wrappers for legacy imports

COMPATIBILITY:
- ✅ NO BREAKING CHANGES - All existing notebook imports work
- Old imports (ml_lib, audio_lib) still functional via compatibility layer
- New imports (genai_lib.ml, genai_lib.audio) recommended for new code
- Deprecation warnings deferred to v0.3.0
"

# Merge in main
git checkout main
git merge feature/restructure-packages

# Tag Release
git tag -a v0.2.0 -m "Version 0.2.0 - Package restructuring"

# Push
git push origin main --tags
```

---

## 🎯 Erwartete Vorteile

### ✅ Zero Breaking Changes
- **Alle existierenden Notebooks funktionieren ohne Änderung**
- Compatibility Wrapper für alte Import-Pfade (ml_lib, audio_lib)
- Keine sofortigen Deprecation Warnings
- Sanfter Migrationspfad über mehrere Versionen

### ✅ Klarheit
- Klare Trennung: Stabil (versioniert) vs. Experimentell (nicht versioniert)
- Ein zentrales `pyproject.toml` für das Haupt-Package
- Moderner Python-Packaging-Standard (PEP 621)
- Konsistente Namensgebung (alles unter `genai_lib.*`)

### ✅ Flexibilität
- Optionale Features über extras installierbar
- Basis-Installation bleibt schlank (nur Core-Dependencies)
- Nutzer können wählen, was sie benötigen
- Beide Import-Syntaxen parallel verfügbar

### ✅ Wartbarkeit
- Zentrale Dependency-Verwaltung in pyproject.toml
- Moderne pyproject.toml statt setup.py
- Klare Modul-Organisation (Separation of Concerns)
- Einfachere Versionierung und Release-Management

### ✅ Erweiterbarkeit
- Neue Sub-Module einfach als `genai_lib/xyz/` hinzufügen
- Neue optionale Dependencies als extras definieren
- Klare Struktur für zukünftige Features
- Skalierbar für weitere Kurs-Module

### ✅ Nutzerfreundlichkeit
- Installation bleibt einfach: `pip install git+...`
- Existierende Kurs-Materialien funktionieren weiter
- Neue Materialien profitieren von moderner Struktur
- Klare Dokumentation für Migration (optional)

---

## 📝 Checkliste vor Start

- [ ] **Backup erstellt** (`git branch backup/pre-migration`)
- [ ] **Entscheidungen getroffen** (Phase 1 ausgefüllt)
- [ ] **Module-Status geklärt** (Was wird integriert/experimentell)
- [ ] **Tests vorhanden** (Wichtige Funktionen dokumentiert)
- [ ] **Zeit eingeplant** (Migration dauert ~2-4 Stunden)

---

## 🆘 Rollback-Plan

Falls während der Migration Probleme auftreten:

```bash
# Zurück zum Backup-Branch
git checkout backup/pre-migration

# Oder: Einzelne Dateien wiederherstellen
git checkout backup/pre-migration -- setup.py
git checkout backup/pre-migration -- genai_lib/

# Oder: Migration komplett abbrechen
git reset --hard backup/pre-migration
```

---

## 📚 Referenzen

- [PEP 621 – Storing project metadata in pyproject.toml](https://peps.python.org/pep-0621/)
- [Python Packaging User Guide](https://packaging.python.org/)
- [setuptools Documentation](https://setuptools.pypa.io/)

---

**Nächste Schritte:**
1. Entscheidungen in Phase 1 treffen
2. Backup erstellen
3. Migration starten mit Phase 2
