![My Image](https://raw.githubusercontent.com/ralf-42/Image/main/genai-banner-2.jpg)

---
Stand 05.2025


# 1 Einleitung

Die Entwicklung von Applikationen, die auf großen Sprachmodellen (Large Language Models, LLMs) basieren, hat in den letzten Jahren rasant an Fahrt aufgenommen. Für Entwickler stellt sich dabei oft die zentrale Frage nach den richtigen Werkzeugen und Frameworks. Zwei prominente Optionen im Jahr 2025 sind die direkte Nutzung der Programmierschnittstelle (API) von OpenAI und der Einsatz des LangChain-Frameworks, typischerweise mit einem OpenAI-Modell als Backend. Angesichts der Schnelllebigkeit in diesem Technologiefeld ist ein aktueller Vergleich unerlässlich, um fundierte Entscheidungen treffen zu können.

Dieser Bericht stellt die aktuellen Funktionalitäten der direkten OpenAI API und des LangChain-Frameworks (bei Verwendung mit OpenAI-Modellen) gegenüber. Der Fokus liegt auf den Kernaspekten Prompting-Mechanismen, Output-Parsing und dem Aufruf von Sprachmodellen. Ergänzend werden die kritischen Themen Tool/Function Calling sowie Fehlerbehandlung und Debugging beleuchtet, da sie für die praktische Entwicklung von LLM-Applikationen von entscheidender Bedeutung sind. Eine wichtige Neuerung seitens OpenAI, die **Responses API**, wird ebenfalls in die Betrachtung einbezogen, da sie potenziell signifikante Auswirkungen auf den Vergleich und die Art und Weise hat, wie Entwickler mit den Modellen interagieren.



# 2 Überblick

Die folgende Tabelle bietet eine erste, hochrangige Zusammenfassung der Kernunterschiede zwischen der direkten Nutzung der OpenAI API und der Verwendung von LangChain mit einem OpenAI-Backend. Sie dient als Referenzpunkt für die nachfolgenden detaillierten Abschnitte.

|                                             |                                                                                                                            |                                                                                                                                              |
| ------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| **Aspekt**                                  | **OpenAI API (Direkt)**                                                                                                    | **LangChain (mit OpenAI)**                                                                                                                   |
| **Prompting-Mechanismen**                   | `messages`-Array (Chat Completions API) ; `input` & `instructions` (Responses API). Manuelle Strukturierung.               | `PromptTemplate`, `ChatPromptTemplate`. Flexible Template-Erstellung und -Verwaltung. LCEL für Verkettung.                                   |
| **Output-Parsing**                          | Manuelles Parsen von Text; `tool_calls` für strukturiertes JSON.                                                           | Dedizierte `OutputParser` (z.B. `StrOutputParser`, `JsonOutputParser`, `PydanticOutputParser`). Automatische Strukturierung und Validierung. |
| **Aufruf von Sprachmodellen**               | `client.chat.completions.create()`, `client.responses.create()`. Direkter API-Zugriff. Streaming via `stream=True`.        | `ChatOpenAI().invoke()`, `ChatOpenAI().stream()`. Abstrahierte, konsistente Schnittstelle.                                                   |
| **Tool/Function Calling**                   | `tools`-Parameter mit JSON-Schema-Definition. `tool_choice` zur Steuerung. Responses API mit Built-in Tools.               | `bind_tools` mit Pydantic-Modellen oder Funktionen. Orchestrierung innerhalb von Agents.                                                     |
| **Abstraktionsebene**                       | Gering. Direkte Kontrolle über API-Parameter.                                                                              | Hoch. Vereinfachung durch Abstraktionen und standardisierte Komponenten.                                                                     |
| **Flexibilität (Modellagnostik)**           | Bindung an OpenAI-Modelle.                                                                                                 | Hoch. Prinzipiell einfacher Wechsel des LLM-Providers möglich.                                                                               |
| **Fehlerbehandlung & Debugging**            | Standard API-Fehlercodes und Exceptions. Manuelle Implementierung von Retries.                                             | Erweiterte Parser-Retries (`RetryOutputParser`). Observability-Plattform LangSmith für Tracing und Debugging.                                |
| **Zustandsmanagement (Conversation State)** | Manuell mit Chat Completions API (gesamter Verlauf muss gesendet werden).1 Serverseitig mit Responses API (`store: true`). | LangChain Memory-Module für verschiedene Speicherstrategien.                                                                                 |
| **Komplexität der Implementierung**         | Geeignet für einfache, direkte Aufgaben. Potenziell geringere Einstiegshürde für Basisfunktionen.                          | Geeignet für komplexe, mehrstufige Workflows (Chains, Agents). Kann bei einfachen Aufgaben Overhead erzeugen.                                |

Diese Tabelle verdeutlicht, dass die direkte OpenAI API maximale Kontrolle und Direktheit bietet, während LangChain durch Abstraktion, Modularität und spezialisierte Werkzeuge die Entwicklung komplexerer Anwendungen vereinfachen kann.

# 3 Prompting-Mechanismen

Die Art und Weise, wie Anweisungen (Prompts) an ein Sprachmodell formuliert und übergeben werden, ist fundamental für dessen Verhalten und die Qualität der generierten Antworten.

## 3.1 OpenAI API (Direkt)

Bei der direkten Nutzung der OpenAI API erfolgt das Prompting für Chat-Modelle primär über den messages-Parameter. Dieser ist ein Array von Dictionaries, wobei jedes Dictionary eine einzelne Nachricht in der Konversation repräsentiert und die Schlüssel role und content enthält.

Die möglichen Rollen sind:

- `system`: Definiert das übergeordnete Verhalten, die Persönlichkeit oder spezifische Anweisungen für das Modell. Diese Nachricht wird typischerweise am Anfang des `messages`-Arrays platziert.
- `user`: Repräsentiert die Eingaben des Endnutzers oder der Applikation.
- `assistant`: Stellt die vorherigen Antworten des Modells dar.

Für die **Chat Completions API** ist es notwendig, den gesamten bisherigen Konversationsverlauf bei jedem API-Aufruf mitzusenden, damit das Modell den Kontext beibehält. Dies kann bei sehr langen Konversationen zu einer erheblichen Menge an übertragenen Daten führen.

Eine neuere Entwicklung ist die **OpenAI Responses API** (Stand Q1/2025), die alternative Mechanismen bietet. Hier kann der `instructions`-Parameter genutzt werden, um eine Systemnachricht zu übergeben. Dies ist besonders relevant in Kombination mit `previous_response_id`, da die Instruktionen aus einer vorherigen Antwort nicht automatisch auf die nächste übertragen werden, was das flexible Austauschen von Systemanweisungen vereinfacht. Zudem ermöglicht die Responses API mit dem Parameter `store: true` ein serverseitiges Management des Konversationszustands, wodurch das wiederholte Senden des gesamten Verlaufs entfällt.

**Codebeispiel (Python, 2025-kompatibel): OpenAI API Chat Completion**


```Python
from openai import OpenAI

# Es wird davon ausgegangen, dass der OPENAI_API_KEY als Umgebungsvariable gesetzt ist.
client = OpenAI()

try:
    chat_completion = client.chat.completions.create(
        model="gpt-4o", # Stand 2025 ein gängiges, leistungsfähiges Modell
        messages=[
            {"role": "system", "content": "Du bist ein hilfreicher Assistent, der prägnante Antworten gibt."},
            {"role": "user", "content": "Was ist die Hauptstadt von Frankreich?"}
        ]
    )
    if chat_completion.choices:
        assistant_message = chat_completion.choices.message.content
        print(f"OpenAI API Assistent: {assistant_message}")
    else:
        print("Keine Antwort von der API erhalten.")

except Exception as e:
    print(f"Ein Fehler ist aufgetreten: {e}")

# Beispielhafter Aufruf mit der neueren Responses API (Konzept, API kann variieren)
# try:
#     response_stream = client.responses.create(
#         model="gpt-4.1", # Beispielmodell für Responses API
#         input=[
#             {"role": "user", "content": "Erzähle einen Witz über Programmieren."}
#         ],
#         instructions="Antworte humorvoll und kurz.",
#         store=True # Aktiviert serverseitiges Zustandsmanagement
#     )
#     # Verarbeitung des Streams oder der finalen Antwort hier
#     # for event in response_stream:
#     #     if event.type == 'response.output_text.delta':
#     #         print(event.data.delta, end="")
#     print("\n(Beispiel für Responses API)")
# except Exception as e:
#     print(f"Ein Fehler mit der Responses API ist aufgetreten: {e}")

```



## 3.2 LangChain (mit OpenAI)

LangChain abstrahiert die direkte API-Interaktion und bietet für das Prompting mächtige Werkzeuge in Form von `PromptTemplate` und `ChatPromptTemplate`.

- `PromptTemplate` wird für einfache String-basierte Prompts verwendet.
- `ChatPromptTemplate` ist für Chat-Modelle konzipiert und erlaubt die Definition einer Sequenz von Nachrichten, ähnlich dem `messages`-Array der OpenAI API, jedoch mit der Möglichkeit, Platzhalter für dynamische Inhalte zu verwenden.

LangChain unterstützt verschiedene Nachrichtentypen innerhalb eines `ChatPromptTemplate`, darunter `SystemMessage`, `HumanMessage`, `AIMessage` und das flexiblere `ChatMessagePromptTemplate`, das beliebige Rollen erlaubt. Ein besonders nützliches Element ist `MessagesPlaceholder`. Dieser Platzhalter ermöglicht es, eine dynamische Liste von Nachrichten – beispielsweise eine Chat-Historie aus einem Memory-Modul – an einer bestimmten Stelle in das Template einzufügen.

Die erstellten Prompt-Templates werden typischerweise mithilfe der LangChain Expression Language (LCEL) durch den `|` (Pipe)-Operator mit einem LLM-Modell und optional einem Output-Parser verkettet.

**Codebeispiel (Python, 2025-kompatibel): LangChain ChatPromptTemplate**


```Python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage

# Es wird davon ausgegangen, dass der OPENAI_API_KEY als Umgebungsvariable gesetzt ist.
llm = ChatOpenAI(model="gpt-4o", temperature=0) # Stand 2025

# Definition eines ChatPromptTemplate
prompt_template = ChatPromptTemplate.from_messages()

# Alternative Definition mit Tupeln (oft kürzer)
# prompt_template_alt = ChatPromptTemplate.from_messages()

# Verkettung mit LCEL
chain = prompt_template | llm | StrOutputParser()

try:
    # Aufruf der Kette mit dynamischen Werten für die Platzhalter
    response = chain.invoke({"land": "Italien"})
    print(f"LangChain Assistent: {response}")
except Exception as e:
    print(f"Ein Fehler mit LangChain ist aufgetreten: {e}")

```



## 3.3 Vergleichende Analyse

Die direkte OpenAI API bietet granulare Kontrolle über die an das Modell gesendeten Nachrichten. Dies erfordert jedoch ein manuelles Management der Nachrichtenliste und des Konversationskontextes, es sei denn, die neuere Responses API mit serverseitigem State-Management wird genutzt.1

LangChain hingegen führt eine Abstraktionsebene durch Prompt-Templates ein. Diese erhöhen die Wiederverwendbarkeit und Lesbarkeit von Code, insbesondere bei komplexen oder sich häufig ändernden Prompt-Strukturen. Die LCEL ermöglicht eine elegante Verkettung von Komponenten.

Die Wahl des Ansatzes hängt stark von den Projektanforderungen ab. Für einfache Anwendungen mit statischen Prompts mag die direkte API-Nutzung ausreichend sein. Sobald jedoch dynamische Prompt-Generierung, komplexe Kontextualisierung oder die Notwendigkeit zur einfachen Integration von Chat-Historien (über `MessagesPlaceholder`) ins Spiel kommen, bieten die LangChain-Mechanismen deutliche Vorteile in Bezug auf Organisation und Wartbarkeit.

Ein wichtiger Aspekt ist die Entwicklung bei OpenAI selbst. Mit der Einführung der Responses API und Features wie `store: true` für das Zustandsmanagement 1 nähert sich OpenAI Funktionalitäten an, die traditionell Stärken von Frameworks wie LangChain waren. Für Anwendungen, die primär auf OpenAI-Modelle setzen und grundlegendes Konversationsmanagement benötigen, könnte die direkte Nutzung der Responses API eine schlankere Alternative darstellen. LangChain behält seine Stärken bei sehr komplexen Prompt-Manipulationen, der Notwendigkeit von Modellagnostik oder wenn umfangreiche Logik vor dem eigentlichen LLM-Aufruf ausgeführt werden muss. Die Flexibilität von LangChain durch Templates und Platzhalter kann jedoch für Einsteiger eine höhere initiale Lernkurve bedeuten als das direkte Arbeiten mit dem `messages`-Array der OpenAI API.20

# 4 Output-Parsing

Nachdem das LLM eine Antwort generiert hat, ist der nächste entscheidende Schritt das Parsen dieses Outputs, insbesondere wenn strukturierte Daten erwartet werden.

## 4.1 OpenAI API (Direkt)

Standardmäßig liefert die OpenAI API die Antworten der LLMs als reinen Text. Wenn die Anwendung strukturierte Daten, beispielsweise im JSON-Format, benötigt, muss dieser Text manuell geparst und validiert werden.

Der empfohlene und robustere Weg, um strukturierte Daten von der API zu erhalten, ist die Nutzung von **Tool Calls** (früher Function Calling).8 Dabei wird dem Modell im API-Aufruf eine Liste von verfügbaren "Tools" (Funktionen) mitsamt einer Beschreibung und einem JSON-Schema für deren erwartete Parameter übergeben. Das Modell kann dann entscheiden, ein solches Tool "aufzurufen", indem es ein JSON-Objekt zurückgibt, das den Namen des Tools und die zugehörigen Argumente enthält. Dieses JSON-Objekt ist dann bereits strukturiert und kann direkt weiterverarbeitet werden. Die `finish_reason` in der API-Antwort (z.B. `tool_calls`) signalisiert, dass das Modell ein Tool verwenden möchte.2

Die neue **Responses API** ist explizit darauf ausgelegt, Workflows mit Tool-Nutzung zu vereinfachen.1 OpenAI bietet auch einen "strict mode" für Tool Calls, der die Einhaltung des definierten Schemas erzwingen soll, allerdings mit einigen Einschränkungen verbunden ist.8

**Codebeispiel (Python, 2025-kompatibel): OpenAI API Output-Parsing mit Tool Call**


```Python
import json
from openai import OpenAI

client = OpenAI()

# Definition des Tools, das das Modell "aufrufen" kann
tools =
        }
    }
]

messages = [
    {"role": "user", "content": "Max Mustermann ist 30 Jahre alt."}
]

try:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        tool_choice="auto" # Lässt das Modell entscheiden, ob ein Tool genutzt wird
    )

    response_message = response.choices.message
    tool_calls = response_message.tool_calls

    if tool_calls:
        for tool_call in tool_calls:
            if tool_call.name == "extract_user_info":
                function_args = json.loads(tool_call.arguments)
                name = function_args.get("name")
                age = function_args.get("age")
                print(f"OpenAI API (Tool Call) - Name: {name}, Alter: {age}")
    else:
        # Fallback, falls kein Tool Call erfolgte, sondern eine direkte Antwort
        print(f"OpenAI API (Direkte Antwort): {response_message.content}")

except Exception as e:
    print(f"Ein Fehler ist aufgetreten: {e}")
```



## 4.2 LangChain (mit OpenAI)

LangChain stellt eine Vielzahl von `OutputParser`-Klassen zur Verfügung, um die Antworten von LLMs zu verarbeiten und in gewünschte Formate zu überführen.5 Zu den wichtigsten gehören:

- `StrOutputParser`: Gibt die LLM-Antwort als einfachen String zurück. Dies ist die grundlegendste Form des Parsings.12
- `JsonOutputParser`: Versucht, die LLM-Antwort als JSON-Objekt zu parsen. Optional kann ein Pydantic-Modell übergeben werden, um das Schema zu definieren und die geparsten Daten zu validieren.10
- `PydanticOutputParser`: Parst die LLM-Antwort direkt in eine Instanz eines vordefinierten Pydantic-Modells. Dies bietet starke Typsicherheit und eine klare Datenstruktur.10

Viele Parser in LangChain bieten eine Methode `get_format_instructions()`. Diese generiert eine textuelle Anweisung, die dem Prompt hinzugefügt werden kann, um das LLM anzuleiten, seine Ausgabe im korrekten, vom Parser erwarteten Format zu generieren.10 Output-Parser werden typischerweise als letztes Glied in einer LCEL-Kette verwendet: `prompt | model | parser`.10 LangChain unterstützt auch das Streaming von strukturierten Outputs, was bedeutet, dass Teile der strukturierten Antwort bereits verarbeitet werden können, während das Modell noch generiert.10

**Codebeispiel (Python, 2025-kompatibel): LangChain mit PydanticOutputParser**


```Python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field # Wichtig: langchain_core.pydantic_v1 für Kompatibilität
from langchain_core.output_parsers import PydanticOutputParser

# Es wird davon ausgegangen, dass der OPENAI_API_KEY als Umgebungsvariable gesetzt ist.
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Definition des Pydantic-Modells für die strukturierte Ausgabe
class UserInfo(BaseModel):
    name: str = Field(description="Der vollständige Name der Person")
    age: int = Field(description="Das Alter der Person in Jahren")
    city: str = Field(description="Die Stadt, in der die Person wohnt")

# Initialisierung des Parsers mit dem Pydantic-Modell
pydantic_parser = PydanticOutputParser(pydantic_object=UserInfo)

# Erstellung des Prompt-Templates mit Formatierungsanweisungen vom Parser
prompt = PromptTemplate(
    template="Extrahiere die Informationen aus der folgenden Nutzereingabe.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": pydantic_parser.get_format_instructions()}
)

# Verkettung mit LCEL
chain = prompt | llm | pydantic_parser

try:
    user_query = "Anna Schmidt ist 28 Jahre alt und lebt in Berlin."
    parsed_output = chain.invoke({"query": user_query})
    print(f"LangChain (PydanticOutputParser) - Name: {parsed_output.name}, Alter: {parsed_output.age}, Stadt: {parsed_output.city}")
except Exception as e:
    print(f"Ein Fehler mit LangChain ist aufgetreten: {e}")
```



## 4.3 Vergleichende Analyse

Die direkte Nutzung der OpenAI API erfordert für strukturierte Daten entweder manuelles Parsen von Text-Antworten oder die Implementierung des Tool Calling Mechanismus. Letzterer ist der robustere Ansatz, da das Modell angeleitet wird, direkt JSON-konforme Argumente zu liefern.

LangChain bietet mit seinen `OutputParser`-Klassen eine höhere Abstraktionsebene und spezialisierte Werkzeuge, die das Parsen, Strukturieren und Validieren von LLM-Antworten erheblich vereinfachen. Insbesondere die Kombination mit Pydantic-Modellen ermöglicht eine typsichere und gut definierte Verarbeitung strukturierter Daten. Die automatische Generierung von Formatierungsanweisungen hilft zudem, die LLM-Ausgabe in die gewünschte Form zu lenken.

Für Anwendungen, die stark auf komplexe, strukturierte Daten angewiesen sind, bietet LangChain oft einen komfortableren und fehlertoleranteren Entwicklungsprozess. Die Qualität der `format_instructions` bei LangChain-Parsern oder der `description` bei OpenAI-Tools ist dabei entscheidend für die Zuverlässigkeit des LLMs, korrekt formatierte Daten zu liefern.8 Ungenaue Instruktionen können zu Parsing-Fehlern führen. Die Wahl des Mechanismus beeinflusst somit direkt die Robustheit der Anwendung. Während LangChains Parser eine eingebaute Validierungslogik bieten, liegt bei direkter API-Nutzung die Verantwortung für die Validierung (z.B. von Tool Call Argumenten, die nicht immer valides JSON sein müssen 9) und die Fehlerbehandlung stärker beim Entwickler.

Der Trend geht klar in Richtung zuverlässiger Extraktion strukturierter Daten. OpenAI forciert dies über `tool_calls` und die Vereinfachung der Tool-Nutzung in der neuen Responses API.1 LangChain bietet hierfür eine breitere Palette an spezialisierten Parsern.

# 5 Aufruf von Sprachmodellen (LLM)

Der technische Aufruf des Sprachmodells, inklusive der Handhabung von Parametern wie Streaming, unterscheidet sich ebenfalls zwischen der direkten API-Nutzung und LangChain.

## 5.1 OpenAI API (Direkt)

Für Chat-Modelle ist die primäre Methode zum Aufruf des LLMs `client.chat.completions.create()`.2 Mit der Einführung der **Responses API** (Stand Q1/2025) steht zusätzlich `client.responses.create()` zur Verfügung, welche insbesondere für komplexere Workflows mit Tool-Nutzung, Code-Ausführung und Zustandsmanagement konzipiert wurde.1

Das Streaming von Antworten, also das empfangen der generierten Tokens in Echtzeit, wird durch Setzen des Parameters `stream=True` im API-Request aktiviert.4 Die Antwort wird dann als eine Serie von Events oder Chunks geliefert, die iterativ verarbeitet werden können.

Es ist wichtig zu beachten, dass ältere API-Endpunkte wie die `Completions API` (genutzt für Modelle wie `gpt-3.5-turbo-instruct`) zunehmend an Bedeutung verlieren und durch die `Chat Completions API` sowie die neue `Responses API` ersetzt werden. OpenAI fokussiert sich auf einen "conversation-in and message-out" Ansatz 2, und einige ältere Modelle werden sukzessive abgekündigt (deprecated).1

**Codebeispiel (Python, 2025-kompatibel): OpenAI API Streaming**



```Python
from openai import OpenAI

client = OpenAI()

try:
    # Streaming mit der Chat Completions API
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=,
        stream=True,
    )
    print("OpenAI API (Streaming): ", end="")
    for chunk in stream:
        if chunk.choices and chunk.choices.delta and chunk.choices.delta.content:
            print(chunk.choices.delta.content, end="", flush=True)
    print("\n")

    # Konzeptuelles Streaming mit der Responses API (Syntax kann leicht variieren)
    # print("OpenAI Responses API (Streaming): ", end="")
    # response_api_stream = client.responses.create(
    #     model="gpt-4.1", # Beispielmodell für Responses API
    #     input=[{"role": "user", "content": "Zähle bis 3."}],
    #     stream=True
    # )
    # for event in response_api_stream:
    #     if event.type == 'response.output_text.delta': # Beispielhafter Event-Typ
    #         print(event.data.delta, end="", flush=True) # Zugriff auf Delta kann variieren
    # print("\n(Beispiel für Responses API Streaming)")

except Exception as e:
    print(f"Ein Fehler ist aufgetreten: {e}")
```

_Code basierend auf 4_

## 5.2 LangChain (mit OpenAI)

In LangChain dient die Klasse `ChatOpenAI` aus dem Paket `langchain_openai` als Wrapper für die Chat-Modelle von OpenAI.11 Der Aufruf des Modells erfolgt über standardisierte Methoden:

- `invoke()`: Für eine einzelne, vollständige Antwort vom Modell.
- `stream()`: Für eine gestreamte Antwort, bei der die Tokens nacheinander eintreffen.5

Typischerweise wird das `ChatOpenAI`-Objekt mittels LCEL in eine Kette eingebunden, z.B. `prompt | llm | parser`.12 LangChain abstrahiert dabei die direkten API-Aufrufe. Wenn die `stream()`-Methode verwendet wird, setzt LangChain intern die notwendigen Parameter (wie `stream=True` bei der OpenAI API) und liefert einen Iterator über die Antwort-Chunks. LangChain bietet auch die Möglichkeit, die Token-Nutzung über `usage_metadata` in der Antwort zu verfolgen.11

**Codebeispiel (Python, 2025-kompatibel): LangChain Streaming mit ChatOpenAI**



```Python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4o", temperature=0)

prompt = ChatPromptTemplate.from_messages()

# Verkettung mit LCEL für Streaming
chain = prompt | llm # Für StrOutputParser würde man `| StrOutputParser()` anhängen

try:
    print("LangChain (Streaming): ", end="")
    # Die stream() Methode gibt einen Iterator über AIMessageChunk Objekte zurück
    for chunk in chain.stream({}): # Leeres Dict, da Prompt keine Variablen hat
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print("\n")
    
    # Abrufen von usage_metadata nach einem invoke-Aufruf (nicht direkt beim Streamen jedes Chunks)
    # full_response = chain.invoke({})
    # if hasattr(full_response, 'usage_metadata') and full_response.usage_metadata:
    #     print(f"Token-Nutzung: {full_response.usage_metadata}")

except Exception as e:
    print(f"Ein Fehler mit LangChain ist aufgetreten: {e}")
```



## 5.3 Vergleichende Analyse

Die direkte OpenAI API bietet unmittelbaren Zugriff auf die API-Endpunkte und volle Kontrolle über alle Request-Parameter. Die Einführung der Responses API 1 stellt eine signifikante Weiterentwicklung dar, die komplexere Interaktionsmuster nativ unterstützt.

LangChain vereinfacht den Aufruf von LLMs durch eine konsistente Schnittstelle (`invoke()`, `stream()`) und die nahtlose Integration in LCEL-Ketten. Dies reduziert die Komplexität des Codes für den Entwickler, führt aber auch eine zusätzliche Abstraktionsebene ein. Streaming ist bei beiden Ansätzen gut unterstützt und ermöglicht interaktive Anwendungen.

Die Weiterentwicklung der nativen OpenAI API-Primitiven, wie die Responses API, könnte die Notwendigkeit von Frameworks wie LangChain für bestimmte Anwendungsfälle reduzieren, insbesondere wenn die native API bereits "mächtig genug" wird und Aufgaben wie Zustandsmanagement oder vereinfachte Tool-Integration übernimmt.1 Die Abstraktion durch LangChain kann die Entwicklung beschleunigen 20, birgt aber auch das Risiko, dass die Fehlersuche bei unerwartetem Verhalten oder bei Inkompatibilitäten mit neuesten API-Features erschwert wird, falls das Verhalten der Abstraktion nicht gänzlich verstanden ist.39 Für Projekte, die ausschließlich OpenAI-Modelle nutzen, könnte die direkte API-Nutzung, insbesondere mit der Responses API, für viele Szenarien ausreichend und potenziell performanter sein.20 LangChain behält seine Stärken bei Modellagnostik und der Implementierung sehr komplexer, benutzerdefinierter Logikketten.

# 6 Tool/Function Calling
Tool Calling (früher Function Calling) ist ein Mechanismus, der es LLMs ermöglicht, strukturierte Daten zu generieren oder externe Systeme und APIs aufzurufen, um ihre Fähigkeiten zu erweitern.

## 6.1 OpenAI API (Direkt)

Bei der direkten Nutzung der OpenAI API wird der `tools`-Parameter (der den älteren `functions`-Parameter ersetzt) im API-Aufruf verwendet.8 Diesem Parameter wird eine Liste von Tool-Definitionen übergeben. Jede Definition enthält:

- `type`: Muss `"function"` sein.
- `name`: Der Name der Funktion/des Tools.
- `description`: Eine für das LLM verständliche Beschreibung, was das Tool tut und wann es verwendet werden sollte. Klare und detaillierte Beschreibungen sind hier essenziell.8
- `parameters`: Ein JSON-Schema, das die vom Tool erwarteten Eingabeparameter definiert.

Das LLM entscheidet basierend auf dem Prompt und den Tool-Beschreibungen, ob und welches Tool mit welchen Argumenten aufgerufen werden soll. Wenn das Modell ein Tool nutzen möchte, enthält seine Antwort ein `tool_calls`-Objekt (oder mehrere bei parallelen Aufrufen). Dieses Objekt spezifiziert den Namen des Tools und die vom Modell generierten Argumente als JSON-String.8

Mit dem `tool_choice`-Parameter kann das Verhalten des Modells gesteuert werden:

- `"auto"` (Standard): Das Modell entscheidet frei.
- `"required"`: Das Modell muss mindestens ein Tool aufrufen.
- `{"type": "function", "name": "my_function"}`: Das Modell wird gezwungen, genau dieses spezifische Tool aufzurufen.8

Einige neuere Modelle unterstützen parallele Tool-Aufrufe, d.h. das Modell kann in einer einzigen Antwort mehrere Tool-Aufrufe anfordern.9 Der "strict mode" kann für Tool-Aufrufe aktiviert werden, um die Einhaltung des Schemas zu erzwingen, hat aber gewisse Einschränkungen, z.B. müssen alle Felder als `required` markiert und `additionalProperties` auf `false` gesetzt sein.8

Die **OpenAI Responses API** (Stand Q1/2025) integriert Tool-Nutzung weiter und bietet sogar eingebaute Tools wie `web_search_preview`, `file_search` (für RAG-ähnliche Funktionalität) und `computer_use_preview` (für Browser-Interaktionen).1

**Codebeispiel (Python, 2025-kompatibel): OpenAI API Tool Calling**



```Python
import json
from openai import OpenAI

client = OpenAI()

# Definition des Tools
tools_definition =},
            },
            "required": ["location"],
        },
    }
]

messages = [{"role": "user", "content": "Wie ist das Wetter in London in Celsius?"}]

try:
    # Verwendung von client.chat.completions.create
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools_definition,
        tool_choice="auto",  # oder z.B. {"type": "function", "name": "get_current_weather"}
    )
    response_message = response.choices.message

    if response_message.tool_calls:
        for tool_call in response_message.tool_calls:
            if tool_call.name == "get_current_weather":
                function_args = json.loads(tool_call.arguments)
                location = function_args.get("location")
                unit = function_args.get("unit", "celsius") # Default falls nicht spezifiziert
                print(f"OpenAI API - Tool Call: get_current_weather für {location} in {unit}")
                # Hier würde der tatsächliche Aufruf der Wetterfunktion erfolgen
                # z.B. weather_data = get_weather_function(location, unit)
                # Und dann eine neue Nachricht mit den Tool-Ergebnissen an die API senden
    else:
        print(f"OpenAI API - Direkte Antwort: {response_message.content}")

except Exception as e:
    print(f"Ein Fehler ist aufgetreten: {e}")

```



## 6.2 LangChain (mit OpenAI)

LangChain vereinfacht die Arbeit mit Tools durch die Methode `bind_tools` des `ChatOpenAI`-Objekts.5 Diese Methode akzeptiert:

- Pydantic-Modelle: Das Schema der Tool-Parameter wird direkt aus dem Pydantic-Modell abgeleitet.
- Python-Funktionen: LangChain versucht, das Schema aus den Typ-Annotationen der Funktion zu inferieren.
- LangChain `Tool`-Objekte: Eine LangChain-spezifische Abstraktion für Tools.

LangChain konvertiert diese Definitionen intern in das von der OpenAI API erwartete JSON-Schemaformat.12 Wenn das LLM ein Tool aufrufen möchte, enthält die zurückgegebene `AIMessage` ein `tool_calls`-Attribut. Dieses Attribut beinhaltet die vom Modell angeforderten Aufrufe in einem standardisierten, provider-agnostischen Format.12

Die eigentliche Stärke von LangChain liegt hier oft in der nachgelagerten Orchestrierung: die Ausführung der aufgerufenen Tools und die Rückmeldung der Ergebnisse an das LLM. Dies ist typischerweise Teil der Logik von LangChain Agents. LangChain bietet auch Möglichkeiten, das Erzwingen eines Tools oder das Deaktivieren paralleler Tool-Aufrufe zu steuern, falls vom Modell unterstützt.5

**Codebeispiel (Python, 2025-kompatibel): LangChain Tool Calling mit Pydantic**



```Python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import HumanMessage
import json

# Es wird davon ausgegangen, dass der OPENAI_API_KEY als Umgebungsvariable gesetzt ist.
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Definition des Tool-Schemas mit Pydantic
class GetWeatherTool(BaseModel):
    """Ruft das aktuelle Wetter für einen bestimmten Ort ab.""" # Docstring wird als Beschreibung genutzt
    location: str = Field(..., description="Der Ort, z.B. 'Berlin, Deutschland'")
    unit: str = Field(default="celsius", description="Die Temperatureinheit, 'celsius' oder 'fahrenheit'")

# Binden des Tools an das LLM
llm_with_tools = llm.bind_tools()

# Erstellen einer einfachen Kette
prompt = ChatPromptTemplate.from_messages([
    HumanMessage(content="{user_query}")
])
chain = prompt | llm_with_tools

try:
    ai_msg = chain.invoke({"user_query": "Wie ist das Wetter in Paris?"})

    if ai_msg.tool_calls:
        for tool_call in ai_msg.tool_calls:
            # tool_call ist hier ein LangChain ToolCall-Objekt
            # tool_call.name, tool_call.args (dict), tool_call.id
            if tool_call['name'] == "GetWeatherTool": # Name des Pydantic-Modells
                print(f"LangChain - Tool Call: {tool_call['name']} mit Argumenten: {tool_call['args']}")
                # Hier würde die Logik zur Ausführung des Tools stehen
                # und das Ergebnis zurück an das LLM gesendet werden (typischerweise in einem Agenten-Loop)
    else:
        print(f"LangChain - Direkte Antwort: {ai_msg.content}")

except Exception as e:
    print(f"Ein Fehler mit LangChain ist aufgetreten: {e}")

```



## 6.3 Vergleichende Analyse

Beide Ansätze ermöglichen die leistungsstarke Tool Calling Funktionalität. Die direkte OpenAI API gibt dem Entwickler die volle Kontrolle über die JSON-Schema-Definition der Tools.

LangChain vereinfacht die Definition und das Binden von Tools erheblich, insbesondere durch die direkte Nutzung von Pydantic-Klassen oder Python-Funktionen. Die Abstraktion der Schema-Konvertierung kann Entwicklungszeit sparen. Die eigentliche Stärke von LangChain manifestiert sich jedoch oft in der Orchestrierung der Tool-Ausführungen und der komplexen Interaktionslogik, die in Agenten-Systemen benötigt wird.

Tool Calling hat sich als fundamentaler Mechanismus etabliert, um LLMs mit externen Datenquellen und Fähigkeiten auszustatten. OpenAI treibt diese Entwicklung aktiv voran, wie die Integration von Built-in Tools in die Responses API zeigt.1 Die Klarheit und Präzision der Tool-Beschreibungen und Parameterdefinitionen ist für die zuverlässige Funktionsweise entscheidend.8 Vage oder missverständliche Beschreibungen führen dazu, dass das LLM falsche Tools wählt oder inkorrekte Argumente generiert.

Während die OpenAI API die grundlegende Infrastruktur für Tool Calls bereitstellt, kann LangChain den Aufwand für die Integration und Verwaltung multipler Tools, insbesondere in komplexen Agenten-Architekturen, deutlich reduzieren. Die Wahl hängt davon ab, ob eine "Batteries-included"-Lösung für die Tool-Orchestrierung gesucht wird (eher LangChain) oder ob die Logik der Tool-Auswahl und -Ausführung vollständig selbst implementiert werden soll (eher direkte OpenAI API). Die neuen "built-in tools" der OpenAI Responses API 1 könnten jedoch einfachere Anwendungsfälle ohne den Overhead von LangChain abdecken.

<div style="page-break-after: always;"></div>

# 7 Fehlerbehandlung & Debugging

Robuste Fehlerbehandlung und effektive Debugging-Strategien sind unerlässlich für die Entwicklung zuverlässiger LLM-Applikationen.

## 7.1 OpenAI API (Direkt)

Die OpenAI API verwendet Standard-HTTP-Statuscodes, um allgemeine Fehler zu signalisieren (z.B. 4xx für Client-Fehler, 5xx für Server-Fehler). Detailliertere Fehlerinformationen werden im JSON-Body der Fehlerantwort bereitgestellt und enthalten typischerweise Felder wie `type` (z.B. `invalid_request_error`, `api_error`), `code` (ein spezifischerer Fehlercode, z.B. `rate_limit_exceeded`, `model_not_found`) und `message` (eine menschenlesbare Beschreibung des Fehlers).15

Die offizielle OpenAI Python-Bibliothek fängt diese Fehler ab und wirft spezifische Python-Exceptions, die von `openai.APIError` erben, wie z.B. `openai.RateLimitError`, `openai.NotFoundError` oder `openai.BadRequestError` (welches `InvalidRequestError` abgelöst hat).15 Entwickler müssen diese Exceptions in ihrem Code explizit behandeln.

Beim Einsatz von Tool Calling ist es zudem wichtig, die vom Modell generierten JSON-Argumente auf Gültigkeit zu prüfen und potenzielle Fehler beim Parsen oder bei der Tool-Ausführung abzufangen.9 Das Debugging erfolgt primär durch sorgfältiges Logging der API-Requests und -Responses sowie die Analyse der zurückgegebenen Fehlermeldungen.

**Codebeispiel (Python, 2025-kompatibel): OpenAI API Fehlerbehandlung**



```Python
from openai import OpenAI, APIError, RateLimitError, BadRequestError

client = OpenAI()

try:
    chat_completion = client.chat.completions.create(
        model="gpt-4o-hopefully-valid-model", # Beispiel für potenziellen Fehler
        messages=[{"role": "user", "content": "Hallo Welt!"}]
    )
    #... Verarbeitung der Antwort...
    print("OpenAI API: Erfolgreich.")

except RateLimitError as e:
    print(f"OpenAI API RateLimitError: Das Ratenlimit wurde überschritten. Details: {e}")
except BadRequestError as e:
    print(f"OpenAI API BadRequestError: Ungültige Anfrage. Details: {e}")
    # Beinhaltet Fehler wie ungültige Parameter, falsches Modell etc. [15]
except APIError as e:
    print(f"Allgemeiner OpenAI APIError: {e}")
except Exception as e:
    print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")

```

## 7.2 LangChain (mit OpenAI)

LangChain fängt in der Regel Fehler von der zugrundeliegenden API (in diesem Fall der OpenAI API) ab und kann diese entweder direkt weiterleiten oder in eigene, LangChain-spezifische Exceptions umwandeln.

Ein besonderer Vorteil von LangChain liegt in den spezialisierten Output-Parsern, die Mechanismen zur Fehlerkorrektur bieten. Der `RetryOutputParser` und der `OutputFixingParser` können beispielsweise versuchen, Parsing-Fehler automatisch zu beheben, indem sie das LLM mit modifizierten Anweisungen erneut aufrufen.10 Der `RetryOutputParser` ist so konzipiert, dass er den ursprünglichen Prompt und die fehlerhafte Completion erneut an ein (möglicherweise anderes) LLM sendet, mit der Bitte, den Fehler zu korrigieren.16

Für das Debugging und die Observability von LangChain-Anwendungen ist **LangSmith** ein zentrales Werkzeug.5 LangSmith ist eine Plattform, die detailliertes Tracing, Monitoring und Evaluierung von LLM-Applikationen ermöglicht – nicht nur für LangChain, sondern auch für andere Frameworks oder direkte API-Nutzung. Es bietet tiefe Einblicke in jeden Schritt einer Kette oder eines Agentenlaufs, visualisiert Token-Nutzung, Latenzen und aufgetretene Fehler.

Zusätzlich ermöglichen LangChain Callbacks das Einhaken in verschiedene Phasen der Ausführungskette für benutzerdefiniertes Logging oder spezifische Fehlerbehandlungsroutinen.5 LangGraph, eine Erweiterung von LangChain für den Bau zustandsbehafteter, Multi-Akteur-Anwendungen, bietet ebenfalls verbesserte Visualisierungs- und Debugging-Möglichkeiten für komplexe Abläufe.37

**Codebeispiel (Python, 2025-kompatibel): LangChain mit RetryOutputParser**



```Python
from langchain_openai import ChatOpenAI, OpenAI # OpenAI für den Retry-LLM
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser, OutputParserException
from langchain.output_parsers.retry import RetryOutputParser # Import aus langchain.output_parsers

# Es wird davon ausgegangen, dass der OPENAI_API_KEY als Umgebungsvariable gesetzt ist.

# Pydantic Modell für die gewünschte Ausgabe
class Action(BaseModel):
    action: str = Field(description="Die auszuführende Aktion")
    action_input: str = Field(description="Die Eingabe für die Aktion")

# Haupt-LLM und Parser
main_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0) # Ein günstigeres Modell, das eher Fehler macht
parser = PydanticOutputParser(pydantic_object=Action)

prompt = PromptTemplate(
    template="Der Nutzer möchte folgendes tun: {query}.\n{format_instructions}\nAntworte nur mit dem JSON-Objekt.",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# LLM für den RetryOutputParser (kann dasselbe oder ein anderes sein)
retry_llm = OpenAI(temperature=0) # Oft ein einfacheres, günstigeres Modell für Korrekturen

# Erstellen des RetryOutputParser
# Wichtig: Der RetryOutputParser erwartet, dass der `parser` einen String parst.
# Wenn der PydanticOutputParser direkt verwendet wird, kann es zu Typfehlern kommen,
# da PydanticOutputParser ein Pydantic-Objekt erwartet/liefert.
# Eine gängige Lösung ist, eine Kette zu bauen, die sicherstellt, dass der Retry-Mechanismus
# mit String-Repräsentationen arbeitet oder der Parser entsprechend angepasst wird.
# Für dieses Beispiel wird eine vereinfachte Annahme getroffen, dass der Output des LLM
# zunächst als String behandelt wird, bevor der PydanticParser ihn verarbeitet.

# Korrekter Aufbau einer Kette mit RetryOutputParser ist komplexer,
# hier ein konzeptioneller Hinweis auf die Nutzung.
# Die direkte Integration von PydanticOutputParser in RetryOutputParser erfordert oft
# eine sorgfältige Handhabung der Ein- und Ausgabetypen der beteiligten Parser.
# Siehe LangChain Dokumentation für fortgeschrittene Muster.

# Konzeptioneller Einsatz:
# retry_parser = RetryOutputParser.from_llm(parser=parser, llm=retry_llm, prompt=prompt)
# full_chain = prompt | main_llm | retry_parser # Dies kann zu Typfehlern führen, je nach Implementierung

# Ein robusterer Ansatz mit RetryOutputParser involviert oft eine explizite Fehlerbehandlung
# und erneuten Aufruf. Hier ein vereinfachtes Beispiel, das die Idee illustriert:

bad_response_from_llm = '{"action": "search"}' # Fehlendes action_input

try:
    parsed_action = parser.parse(bad_response_from_llm)
    print(f"LangChain - Direkt geparst: {parsed_action}")
except OutputParserException as e:
    print(f"LangChain - Ursprünglicher Parsing-Fehler: {e}")
    # Hier könnte der RetryOutputParser oder OutputFixingParser eingesetzt werden.
    # Beispiel mit OutputFixingParser (ähnlich zu Retry, aber oft einfacher für Pydantic)
    from langchain.output_parsers import OutputFixingParser
    fix_parser = OutputFixingParser.from_llm(parser=parser, llm=ChatOpenAI(model="gpt-4o"))
    try:
        fixed_action = fix_parser.parse(bad_response_from_llm) # OutputFixingParser versucht, den Fehler zu beheben
        print(f"LangChain - Korrigiert durch OutputFixingParser: {fixed_action}")
    except OutputParserException as e_fix:
        print(f"LangChain - Fehler auch nach Korrekturversuch: {e_fix}")

# Hinweis zur LangSmith-Integration (Tracing aktivieren):
# import os
# os.environ = "true"
# os.environ["LANGCHAIN_API_KEY"] = "DEIN_LANGSMITH_API_KEY" # Ersetzen durch eigenen Key
# os.environ = "Mein Projektname" # Optional
# Nun würden alle LangChain Aufrufe automatisch an LangSmith gesendet.
```


## 7.3 Vergleichende Analyse

Die direkte OpenAI API liefert grundlegende Fehlerinformationen, deren Interpretation und Handhabung (z.B. Implementierung von Retry-Logik) dem Entwickler obliegt.

LangChain bietet hier deutlich erweiterte Möglichkeiten. Spezialisierte Parser wie `RetryOutputParser` können die Robustheit gegenüber fehlerhaften LLM-Ausgaben erhöhen, indem sie automatische Korrekturversuche unternehmen. Der entscheidende Vorteil für komplexe Anwendungen ist jedoch die Observability-Plattform LangSmith. Sie ermöglicht ein tiefgreifendes Verständnis des Anwendungsverhaltens, was für das Debugging nicht-deterministischer Systeme, die LLM-Ketten und Agenten beinhalten, unerlässlich ist.

Mit zunehmender Komplexität von LLM-Anwendungen wird eine effektive Fehlerbehandlung und ein leistungsfähiges Debugging immer wichtiger. Reine API-Fehlercodes reichen oft nicht aus, um das Verhalten von Systemen zu verstehen, die aus vielen miteinander verbundenen, potenziell fehleranfälligen Komponenten bestehen (API-Verfügbarkeit, Ratenlimits, fehlerhafte Prompts, falsche Tool-Auswahl, ungültige LLM-Outputs, Fehler in externen Tools). Die "Blackbox"-Natur von LLMs erschwert das Debugging zusätzlich. Werkzeuge wie LangSmith, die ein detailliertes Tracing jedes Schritts ermöglichen 45, sind daher von großem Wert, um Ursachen für unerwartetes Verhalten oder Fehler in komplexen LangChain-Anwendungen zu identifizieren. Ohne solche Werkzeuge ist man oft auf zeitaufwendiges "Print-Debugging" oder Vermutungen angewiesen. Die Investition in das Verständnis und die Nutzung von Observability-Tools kann die Entwicklungszeit signifikant verkürzen und die Zuverlässigkeit von LangChain-Anwendungen maßgeblich verbessern. Bei direkter OpenAI API-Nutzung müssen Entwickler äquivalente Logging- und Debugging-Strategien selbst aufbauen.

<div style="page-break-after: always;"></div>

# 8 Einsatzszenarien und Best Practices

Die Entscheidung zwischen der direkten Nutzung der OpenAI API und dem Einsatz von LangChain hängt stark von den spezifischen Anforderungen des Projekts, der Komplexität der Anwendung und den Präferenzen des Entwicklungsteams ab.

## 8.1 Wann OpenAI API?

- **Einfache, klar definierte Aufgaben:** Wenn die Anwendung eine überschaubare Aufgabe erfüllt, wie z.B. eine einfache Frage-Antwort-Funktion oder Textgenerierung basierend auf einem statischen Prompt, und minimale Latenz sowie volle Kontrolle über den API-Request entscheidend sind.19 Die geringere Anzahl an Abstraktionsebenen kann hier zu einer besseren Performance führen.20
- **Primäre Nutzung von OpenAI-Modellen:** Wenn ausschließlich OpenAI-Modelle zum Einsatz kommen und keine Notwendigkeit für Abstraktionen besteht, die einen Wechsel des LLM-Providers erleichtern würden.13
- **Schnelles Prototyping mit minimalem Setup:** Für sehr schnelle Prototypen oder Experimente, bei denen der Overhead eines zusätzlichen Frameworks vermieden werden soll und die Komplexität gering ist.13
- **Vermeidung von Framework-Abhängigkeiten:** Wenn Entwickler die "Magie" oder die zusätzlichen Abstraktionsebenen von LangChain als hinderlich empfinden oder eine direkte, transparente Kontrolle über jeden Aspekt der API-Interaktion bevorzugen.39
- **Nutzung neuester OpenAI API-Features:** Wenn die neuesten Funktionen der OpenAI API (z.B. die Responses API mit serverseitigem State Management und Built-in Tools 1) die benötigte Funktionalität bereits nativ und zufriedenstellend abdecken.

## 8.2 Wann LangChain?

- **Komplexe Anwendungen:** Für Anwendungen, die komplexe Logik erfordern, wie die Verkettung mehrerer LLM-Aufrufe (Chains), die Implementierung von Agenten-Logik mit Entscheidungsfindung und Tool-Orchestrierung, oder anspruchsvolles Memory-Management für langanhaltende Konversationen.13
- **Retrieval Augmented Generation (RAG):** Wenn RAG ein Kernbestandteil der Anwendung ist, um LLMs mit externem Wissen anzureichern. LangChain bietet hierfür umfassende Komponenten (Document Loaders, Text Splitters, Vector Stores, Retrievers).14
- **Modellagnostik:** Wenn die Anwendung so konzipiert werden soll, dass potenziell verschiedene LLM-Provider (neben OpenAI auch Anthropic, Cohere, lokale Modelle etc.) genutzt oder einfach ausgetauscht werden können.13
- **Fortgeschrittenes Output-Parsing und Fehlerbehandlung:** Wenn robuste Mechanismen für das Parsen strukturierter Daten (z.B. mit Pydantic-Validierung) und automatische Fehlerkorrekturversuche (z.B. `RetryOutputParser`) benötigt werden.10
- **Observability und Debugging mit LangSmith:** Wenn die Nachvollziehbarkeit, das Monitoring und das Debugging komplexer LLM-Ketten und Agenten für die Entwicklung und Wartung essenziell sind. LangSmith ist hier ein mächtiges Werkzeug.17
- **Beschleunigung der Entwicklung:** Durch die Nutzung vorgefertigter Komponenten, Standardisierungen und Abstraktionen kann LangChain die Entwicklungszeit für bestimmte Anwendungsfälle verkürzen.20

## 8.3 Best Practices für beide Ansätze:

- **OpenAI API (Direkt):**
    - **API-Key Management:** API-Keys niemals im Code hardcoden, sondern sicher über Umgebungsvariablen oder Secret-Management-Dienste verwalten.56
    - **Nutzungsüberwachung:** Kosten und API-Nutzung regelmäßig überwachen, um Budgets einzuhalten.56
    - **Prompt Engineering:** Klare, präzise und detaillierte Prompts formulieren. Dies gilt insbesondere für die Beschreibungen von Tools und deren Parametern, um die korrekte Auswahl und Anwendung durch das LLM sicherzustellen.8
    - **Fehlerbehandlung:** Eine robuste Fehlerbehandlung für API-Fehler (Netzwerkprobleme, Ratenlimits, ungültige Anfragen) und fehlerhafte Modellantworten implementieren.9
- **LangChain (mit OpenAI):**
    - **Verständnis der Abstraktionen:** Ein gutes Verständnis dafür entwickeln, wie LangChain-Komponenten intern funktionieren und auf die zugrundeliegende OpenAI API abgebildet werden.
    - **LangSmith nutzen:** Für Debugging, Monitoring und Evaluierung komplexer Ketten und Agenten LangSmith einsetzen.17
    - **Modellspezifisches Prompting:** Auch wenn LangChain eine gewisse Abstraktion bietet, können Prompts dennoch modellspezifische Optimierungen erfordern, um die besten Ergebnisse zu erzielen.57
    - **Iterative Entwicklung:** Prompts, Ketten und Agenten iterativ entwickeln und kontinuierlich testen und verfeinern.57
- **Für beide Ansätze:**
    - **Kostenmanagement:** Mit einfacheren, kostengünstigeren Modellen für das Prototyping beginnen und erst für die Produktion oder anspruchsvollere Aufgaben auf teurere Modelle umsteigen.39
    - **Testing und Validierung:** Die generierten Antworten und das Verhalten der Anwendung kontinuierlich testen und validieren.
    - **Sicherheit:** Sicherheitsaspekte berücksichtigen, insbesondere bei der Verwendung von Tools, die mit externen Systemen interagieren oder Code ausführen (Risiko von Prompt Injection).1

Die Grenze zwischen der "direkten API-Nutzung" und der "Framework-Nutzung" kann verschwimmen, da OpenAI selbst zunehmend High-Level-Funktionen in seine API integriert (z.B. die Responses API 1 oder die Assistants API 14). Dies könnte dazu führen, dass sich Frameworks wie LangChain stärker auf sehr komplexe Orchestrierungsaufgaben, Multi-Modell-Szenarien und spezialisierte Agenten-Architekturen konzentrieren. OpenAI hat ein natürliches Interesse daran, die Nutzung seiner API so einfach und leistungsfähig wie möglich zu gestalten, um Entwickler direkt an die eigene Plattform zu binden. Features, die häufig über Frameworks realisiert werden, könnten daher nativ implementiert werden, wenn eine breite Nachfrage besteht.

Die Wahl zwischen direkter API und LangChain ist oft eine Abwägung zwischen Entwicklungsgeschwindigkeit und Abstraktion auf der einen Seite und Performance, feingranularer Kontrolle und Transparenz auf der anderen Seite.20 LangChain kann die Entwicklungszeit für komplexe Systeme verkürzen 20, fügt aber eine zusätzliche Schicht hinzu, die potenziell Latenz verursachen oder das Debugging ohne Werkzeuge wie LangSmith erschweren kann. Es gibt keine "One-size-fits-all"-Lösung. Die "Best Practice" besteht darin, die Anforderungen des Projekts genau zu analysieren. Für ein Startup, das schnell einen MVP mit OpenAI entwickeln möchte, könnte die direkte API (ggf. unter Nutzung der Responses API) ideal sein. Für ein Unternehmen, das eine komplexe, modellagnostische Agentenplattform mit RAG und umfangreichem Tooling aufbaut, ist LangChain wahrscheinlich die passendere Wahl. Die Möglichkeit, LangChain für die übergeordnete Orchestrierung zu nutzen und dabei spezifische Aufgaben an OpenAI-Agenten (oder die Responses API) zu delegieren, stellt einen interessanten hybriden Ansatz dar.13

# 9 Zusammenfassung 

Der Vergleich zwischen der direkten Nutzung der OpenAI API und dem Einsatz von LangChain (mit OpenAI-Modellen) zeigt, dass beide Ansätze ihre Berechtigung und spezifische Stärken haben, die je nach Anwendungsfall und Entwicklerpräferenz zum Tragen kommen.

**Kernunterschiede und Trade-offs:**

- **Direkte OpenAI API:** Bietet maximale Kontrolle, Transparenz und potenziell geringere Latenz durch weniger Abstraktionsebenen. Der Entwickler interagiert unmittelbar mit den API-Primitiven von OpenAI. Dies erfordert jedoch oft mehr manuellen Code für Aufgaben wie Prompt-Management, Output-Parsing komplexer Strukturen (abseits von Tool Calls), Zustandsmanagement (außer bei Nutzung der neueren Responses API) und Fehlerbehandlung. Die Lernkurve für Basisfunktionalitäten kann flacher sein, aber die Implementierung fortgeschrittener Muster liegt vollständig in der Verantwortung des Entwicklers.
- **LangChain (mit OpenAI):** Stellt eine Abstraktionsebene über der direkten API bereit und bietet eine Fülle von modularen Komponenten, standardisierten Schnittstellen (LCEL) und Werkzeugen (PromptTemplates, OutputParser, Memory-Module, Agents). Dies kann die Entwicklung komplexer Anwendungen beschleunigen, die Wiederverwendbarkeit von Code fördern und die Integration verschiedener LLMs oder Datenquellen erleichtern. Die Observability-Plattform LangSmith ist ein signifikanter Mehrwert für das Debugging und Monitoring. Der Preis für diese Abstraktion kann eine etwas höhere Latenz, eine steilere Lernkurve für das Framework selbst und eine Abhängigkeit von der Aktualität und Stabilität des Frameworks sein.

Die Entscheidung ist somit ein Abwägen zwischen Direktheit und Kontrolle (OpenAI API) versus Abstraktion und Orchestrierungsleistung (LangChain). Es geht um Performance versus Entwicklungsgeschwindigkeit und Einfachheit für simple Aufgaben versus Funktionsumfang für komplexe Systeme.

**Handlungsempfehlungen basierend auf Projektszenarien (Stand 2025):**

1. **Szenario 1: Einfache LLM-Integration, Fokus auf OpenAI, schnelle Prototypen, geringe Komplexität.**
    - **Empfehlung:** Direkte Nutzung der **OpenAI API**.
    - **Begründung:** Für klar definierte, überschaubare Aufgaben ist der direkte Weg oft der effizienteste und performanteste. Die neue **OpenAI Responses API** 1 sollte hier besonders evaluiert werden, da sie Funktionen wie serverseitiges Zustandsmanagement und integrierte Tools bietet, die den Bedarf an externen Frameworks für viele gängige Anwendungsfälle reduzieren können.
2. **Szenario 2: Komplexe Workflows, Agenten-Systeme, Retrieval Augmented Generation (RAG), Bedarf an Modellflexibilität oder fortgeschrittener Orchestrierung.**
    - **Empfehlung:** Einsatz von **LangChain** mit einem OpenAI-Modell als Backend.
    - **Begründung:** LangChain spielt seine Stärken bei der Verkettung von Komponenten (Chains), der Implementierung von Agenten mit Tool-Nutzung und Entscheidungslogik, der Integration von RAG-Pipelines und dem Management von Konversationshistorie aus.18 Die LangChain Expression Language (LCEL) 29, spezialisierte Output-Parser 10 und die Debugging-Möglichkeiten mit LangSmith 45 sind hier wertvolle Werkzeuge. Die prinzipielle Modellagnostik von LangChain bietet zudem Zukunftssicherheit.13
3. **Szenario 3: Lernprojekte, Experimente und schrittweise Komplexitätssteigerung.**
    - **Empfehlung:** Mit der **direkten OpenAI API beginnen**, um ein grundlegendes Verständnis für die Funktionsweise von LLMs, Prompting und API-Interaktionen zu entwickeln. Anschließend **LangChain evaluieren**, um Konzepte wie Chains, Agents und fortgeschrittenes Output-Parsing in einem strukturierten Rahmen zu erkunden und die Abstraktionsvorteile für komplexere Aufgaben kennenzulernen.

**Abschließender Gedanke zur Zukunft:**

Die Landschaft der LLM-APIs und -Frameworks ist extrem dynamisch. OpenAI entwickelt seine APIs kontinuierlich weiter und integriert Funktionalitäten, die zuvor Domänen von Frameworks waren (siehe Responses API 1 und Assistants API 14). Gleichzeitig erweitern Frameworks wie LangChain und dessen Ökosystem (z.B. LangGraph 37) ständig ihren Funktionsumfang und verbessern ihre Werkzeuge. Für Entwickler ist es daher unerlässlich, die aktuellen Entwicklungen aufmerksam zu verfolgen und die Wahl ihrer Werkzeuge regelmäßig zu überprüfen, um die für ihre spezifischen Bedürfnisse am besten geeignete und zukunftssicherste Lösung zu finden. Eine hybride Nutzung, bei der LangChain für die Orchestrierung und die direkte OpenAI API (oder spezifische OpenAI-Agenten-Features) für Kernaufgaben genutzt wird, könnte in vielen Fällen einen optimalen Mittelweg darstellen.13



# 10 Referenzen


1.  OpenAI API: Responses vs. Chat Completions - Simon Willison's Weblog, Zugriff am Mai 6, 2025, [https://simonwillison.net/2025/Mar/11/responses-vs-chat-completions/](https://simonwillison.net/2025/Mar/11/responses-vs-chat-completions/)
2.  Work with chat completion models - Azure OpenAI Service - Learn Microsoft, Zugriff am Mai 6, 2025, [https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/chatgpt](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/chatgpt)
3.  API Reference - OpenAI API, Zugriff am Mai 6, 2025, [https://platform.openai.com/docs/api-reference/chat](https://platform.openai.com/docs/api-reference/chat)
4.  Streaming API responses - OpenAI API, Zugriff am Mai 6, 2025, [https://platform.openai.com/docs/guides/streaming-responses](https://platform.openai.com/docs/guides/streaming-responses)
5.  How-to guides - ️ LangChain, Zugriff am Mai 6, 2025, [https://python.langchain.com/docs/how\_to/](https://python.langchain.com/docs/how_to/)
6.  Quick reference | 🦜️ LangChain, Zugriff am Mai 6, 2025, [https://python.langchain.com/v0.1/docs/modules/model\_io/prompts/quick\_start/](https://python.langchain.com/v0.1/docs/modules/model_io/prompts/quick_start/)
7.  Langchain ChatOpenAI Examples - Restack, Zugriff am Mai 6, 2025, [https://www.restack.io/docs/langchain-knowledge-chatopenai-examples-cat-ai](https://www.restack.io/docs/langchain-knowledge-chatopenai-examples-cat-ai)
8.  Function calling - OpenAI API, Zugriff am Mai 6, 2025, [https://platform.openai.com/docs/guides/function-calling](https://platform.openai.com/docs/guides/function-calling)
9.  How to use function calling with Azure OpenAI Service - Learn Microsoft, Zugriff am Mai 6, 2025, [https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/function-calling](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/function-calling)
10.  A Comprehensive Guide to Output Parsers - Analytics Vidhya, Zugriff am Mai 6, 2025, [https://www.analyticsvidhya.com/blog/2024/11/output-parsers/](https://www.analyticsvidhya.com/blog/2024/11/output-parsers/)
11.  ChatOpenAI — LangChain documentation, Zugriff am Mai 6, 2025, [https://python.langchain.com/api\_reference/openai/chat\_models/langchain\_openai.chat\_models.base.ChatOpenAI.html](https://python.langchain.com/api_reference/openai/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html)
12.  ChatOpenAI | 🦜️ LangChain, Zugriff am Mai 6, 2025, [https://python.langchain.com/v0.1/docs/integrations/chat/openai/](https://python.langchain.com/v0.1/docs/integrations/chat/openai/)
13.  LangChain, OpenAI Agents, and the Rise of the Agentic Stack - FullStack Labs, Zugriff am Mai 6, 2025, [https://www.fullstack.com/labs/resources/blog/langchain-openai-agents-and-the-agentic-stack](https://www.fullstack.com/labs/resources/blog/langchain-openai-agents-and-the-agentic-stack)
14.  Assistants API vs using LangChain (or other) library - OpenAI Developer Forum, Zugriff am Mai 6, 2025, [https://community.openai.com/t/assistants-api-vs-using-langchain-or-other-library/956187](https://community.openai.com/t/assistants-api-vs-using-langchain-or-other-library/956187)
15.  Error code for OpenAI Chat Completion - API, Zugriff am Mai 6, 2025, [https://community.openai.com/t/error-code-for-openai-chat-completion/1102402](https://community.openai.com/t/error-code-for-openai-chat-completion/1102402)
16.  RetryOutputParser — LangChain documentation, Zugriff am Mai 6, 2025, [https://api.python.langchain.com/en/latest/langchain/output\_parsers/langchain.output\_parsers.retry.RetryOutputParser.html](https://api.python.langchain.com/en/latest/langchain/output_parsers/langchain.output_parsers.retry.RetryOutputParser.html)
17.  An Introduction to Debugging And Testing LLMs in LangSmith - DataCamp, Zugriff am Mai 6, 2025, [https://www.datacamp.com/tutorial/introduction-to-langsmith](https://www.datacamp.com/tutorial/introduction-to-langsmith)
18.  Conceptual guide | 🦜️ LangChain, Zugriff am Mai 6, 2025, [https://python.langchain.com/docs/concepts/](https://python.langchain.com/docs/concepts/)
19.  LangChain vs OpenAI API – Which One Should You Choose for AI Chatbots?, Zugriff am Mai 6, 2025, [https://searchcreators.org/search\_blog/post/langchain-vs-openai-api-which-one-should-you-choos/](https://searchcreators.org/search_blog/post/langchain-vs-openai-api-which-one-should-you-choos/)
20.  LangChain vs OpenAI API: When Simplicity Meets Scalability | Aditya Bhattacharya, Zugriff am Mai 6, 2025, [https://blogs.adityabh.is-a.dev/posts/langchain-vs-openai-simplicity-vs-scalability/](https://blogs.adityabh.is-a.dev/posts/langchain-vs-openai-simplicity-vs-scalability/)
21.  langchain · PyPI, Zugriff am Mai 6, 2025, [https://pypi.org/project/langchain/](https://pypi.org/project/langchain/)
22.  Prompt Templates | 🦜️ LangChain, Zugriff am Mai 6, 2025, [https://python.langchain.com/docs/concepts/prompt\_templates/](https://python.langchain.com/docs/concepts/prompt_templates/)
23.  langchain\_core.prompts.chat.ChatPromptTemplate — LangChain 0.2.17, Zugriff am Mai 6, 2025, [https://api.python.langchain.com/en/latest/prompts/langchain\_core.prompts.chat.ChatPromptTemplate.html](https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html)
24.  ChatPromptTemplate — LangChain documentation, Zugriff am Mai 6, 2025, [https://python.langchain.com/v0.2/api\_reference/core/prompts/langchain\_core.prompts.chat.ChatPromptTemplate.html](https://python.langchain.com/v0.2/api_reference/core/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html)
25.  langchain.prompts.chat.ChatPromptTemplate, Zugriff am Mai 6, 2025, [https://sj-langchain.readthedocs.io/en/latest/prompts/langchain.prompts.chat.ChatPromptTemplate.html](https://sj-langchain.readthedocs.io/en/latest/prompts/langchain.prompts.chat.ChatPromptTemplate.html)
26.  LangChain Expression Language Explained - Pinecone, Zugriff am Mai 6, 2025, [https://www.pinecone.io/learn/series/langchain/langchain-expression-language/](https://www.pinecone.io/learn/series/langchain/langchain-expression-language/)
27.  11-langchain-expression-language.ipynb - GitHub, Zugriff am Mai 6, 2025, [https://github.com/pinecone-io/examples/blob/master/learn/generation/langchain/handbook/11-langchain-expression-language.ipynb](https://github.com/pinecone-io/examples/blob/master/learn/generation/langchain/handbook/11-langchain-expression-language.ipynb)
28.  LangChain Expression Language (LCEL), Zugriff am Mai 6, 2025, [https://js.langchain.com/docs/concepts/lcel/](https://js.langchain.com/docs/concepts/lcel/)
29.  LangChain Expression Language (LCEL), Zugriff am Mai 6, 2025, [https://python.langchain.com/docs/concepts/lcel/](https://python.langchain.com/docs/concepts/lcel/)
30.  LangChain Expression Language (LCEL), Zugriff am Mai 6, 2025, [https://python.langchain.com/v0.1/docs/expression\_language/](https://python.langchain.com/v0.1/docs/expression_language/)
31.  How to parse JSON output | 🦜️ LangChain, Zugriff am Mai 6, 2025, [https://python.langchain.com/docs/how\_to/output\_parser\_json/](https://python.langchain.com/docs/how_to/output_parser_json/)
32.  PydanticOutputParser — LangChain documentation, Zugriff am Mai 6, 2025, [https://python.langchain.com/api\_reference/core/output\_parsers/langchain\_core.output\_parsers.pydantic.PydanticOutputParser.html](https://python.langchain.com/api_reference/core/output_parsers/langchain_core.output_parsers.pydantic.PydanticOutputParser.html)
33.  langchain\_core.output\_parsers.pydantic.PydanticOutputParser — LangChain 0.2.17, Zugriff am Mai 6, 2025, [https://api.python.langchain.com/en/latest/output\_parsers/langchain\_core.output\_parsers.pydantic.PydanticOutputParser.html](https://api.python.langchain.com/en/latest/output_parsers/langchain_core.output_parsers.pydantic.PydanticOutputParser.html)
34.  Pydantic parser - ️ LangChain, Zugriff am Mai 6, 2025, [https://python.langchain.com/v0.1/docs/modules/model\_io/output\_parsers/types/pydantic/](https://python.langchain.com/v0.1/docs/modules/model_io/output_parsers/types/pydantic/)
35.  openai-python/examples/streaming.py at main - GitHub, Zugriff am Mai 6, 2025, [https://github.com/openai/openai-python/blob/main/examples/streaming.py](https://github.com/openai/openai-python/blob/main/examples/streaming.py)
36.  Understanding which models are available to call with the APIs - OpenAI Developer Forum, Zugriff am Mai 6, 2025, [https://community.openai.com/t/understanding-which-models-are-available-to-call-with-the-apis/1141192](https://community.openai.com/t/understanding-which-models-are-available-to-call-with-the-apis/1141192)
37.  Introduction | 🦜️ LangChain, Zugriff am Mai 6, 2025, [https://python.langchain.com/docs/introduction/](https://python.langchain.com/docs/introduction/)
38.  ChatOpenAI — LangChain documentation, Zugriff am Mai 6, 2025, [https://api.python.langchain.com/en/latest/openai/chat\_models/langchain\_openai.chat\_models.base.ChatOpenAI.html](https://api.python.langchain.com/en/latest/openai/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html)
39.  Direct OpenAI API vs. LangChain: A Performance and Workflow Comparison - Reddit, Zugriff am Mai 6, 2025, [https://www.reddit.com/r/LangChain/comments/1hd6w5x/direct\_openai\_api\_vs\_langchain\_a\_performance\_and/](https://www.reddit.com/r/LangChain/comments/1hd6w5x/direct_openai_api_vs_langchain_a_performance_and/)
40.  Direct OpenAI API vs. LangChain: A Performance and Workflow Comparison - Reddit, Zugriff am Mai 6, 2025, [https://www.reddit.com/r/OpenAI/comments/1hd6x21/direct\_openai\_api\_vs\_langchain\_a\_performance\_and/](https://www.reddit.com/r/OpenAI/comments/1hd6x21/direct_openai_api_vs_langchain_a_performance_and/)
41.  Prompting Best Practices for Tool Use (Function Calling) - OpenAI Developer Forum, Zugriff am Mai 6, 2025, [https://community.openai.com/t/prompting-best-practices-for-tool-use-function-calling/1123036](https://community.openai.com/t/prompting-best-practices-for-tool-use-function-calling/1123036)
42.  How to retry when a parsing error occurs - ️ LangChain, Zugriff am Mai 6, 2025, [https://python.langchain.com/docs/how\_to/output\_parser\_retry/](https://python.langchain.com/docs/how_to/output_parser_retry/)
43.  OutputFixingParser — LangChain 0.0.149 - Read the Docs, Zugriff am Mai 6, 2025, [https://lagnchain.readthedocs.io/en/stable/modules/prompts/output\_parsers/examples/output\_fixing\_parser.html](https://lagnchain.readthedocs.io/en/stable/modules/prompts/output_parsers/examples/output_fixing_parser.html)
44.  Langchain vs Langsmith: Framework Comparison + Alternatives | Generative AI Collaboration Platform - Orq.ai, Zugriff am Mai 6, 2025, [https://orq.ai/blog/langchain-vs-langsmith](https://orq.ai/blog/langchain-vs-langsmith)
45.  LangSmith - LangChain, Zugriff am Mai 6, 2025, [https://www.langchain.com/langsmith](https://www.langchain.com/langsmith)
46.  Breaking Down Langchain vs Langsmith for Smarter AI App Building - Lamatic.ai Labs, Zugriff am Mai 6, 2025, [https://blog.lamatic.ai/guides/langchain-vs-langsmith/](https://blog.lamatic.ai/guides/langchain-vs-langsmith/)
47.  LangChain Debug Guide — Restack, Zugriff am Mai 6, 2025, [https://www.restack.io/docs/langchain-knowledge-langchain-debug-guide](https://www.restack.io/docs/langchain-knowledge-langchain-debug-guide)
48.  LangChain, Zugriff am Mai 6, 2025, [https://www.langchain.com/](https://www.langchain.com/)
49.  Revolutionizing AI with LangChain, LangGraph, LangSmith - MyScale, Zugriff am Mai 6, 2025, [https://myscale.com/blog/langchain-langgraph-langsmith-game-changers-now/](https://myscale.com/blog/langchain-langgraph-langsmith-game-changers-now/)
50.  LangSmith Pricing - LangChain, Zugriff am Mai 6, 2025, [https://www.langchain.com/pricing-langsmith](https://www.langchain.com/pricing-langsmith)
51.  Announcing LangSmith, a unified platform for debugging, testing, evaluating, and monitoring your LLM applications - LangChain Blog, Zugriff am Mai 6, 2025, [https://blog.langchain.dev/announcing-langsmith/](https://blog.langchain.dev/announcing-langsmith/)
52.  Spoke to 22 LangGraph devs and here's what we found : r/LangChain - Reddit, Zugriff am Mai 6, 2025, [https://www.reddit.com/r/LangChain/comments/1eh0ly3/spoke\_to\_22\_langgraph\_devs\_and\_heres\_what\_we\_found/](https://www.reddit.com/r/LangChain/comments/1eh0ly3/spoke_to_22_langgraph_devs_and_heres_what_we_found/)
53.  \`RetryOutputParser\` error when used with \`PydanticOutputParser\` · Issue #19145 - GitHub, Zugriff am Mai 6, 2025, [https://github.com/langchain-ai/langchain/issues/19145](https://github.com/langchain-ai/langchain/issues/19145)
54.  Why use Langchain Libraries for instead of OpenAI Libraries? - YouTube, Zugriff am Mai 6, 2025, [https://www.youtube.com/watch?v=g-VHv\_IpbTM](https://www.youtube.com/watch?v=g-VHv_IpbTM)
55.  Switching from Direct calls to OpenAI to Langchain - DevTools daily, Zugriff am Mai 6, 2025, [https://www.devtoolsdaily.com/blog/switching-from-direct-openai-to-langchain/](https://www.devtoolsdaily.com/blog/switching-from-direct-openai-to-langchain/)
56.  Production best practices - OpenAI API, Zugriff am Mai 6, 2025, [https://platform.openai.com/docs/guides/production-best-practices](https://platform.openai.com/docs/guides/production-best-practices)
57.  Langchain Vs Openai Api Comparison - Restack, Zugriff am Mai 6, 2025, [https://www.restack.io/docs/langchain-knowledge-langchain-vs-openai-cat-ai](https://www.restack.io/docs/langchain-knowledge-langchain-vs-openai-cat-ai)