import openai
from flask import Flask, render_template, request, Response, jsonify
import threading, queue, time
import os
from dotenv import load_dotenv
from openai import OpenAI

# .env-Datei laden
load_dotenv()

# API-Key aus der Umgebung laden
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

app = Flask(__name__)

# Globale Variablen für den Fortschritts-Log und die finalen Ergebnisse
progress_queue = queue.Queue()
final_results = None
processing_done = False

def log_message(message):
    """Schreibt eine Nachricht in die globale Fortschritts-Queue.
       Lange Nachrichten werden zeilenweise hinzugefügt."""
    print(message)  # Optional: Ausgabe in der Konsole
    # Zerlege die Nachricht in einzelne Zeilen und füge diese einzeln ein
    for line in message.splitlines():
        progress_queue.put(line)

def call_chat_completion(messages, model="gpt-4o-mini", temperature=0.7, max_completion_tokens=5000):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens
    )
    return response.choices[0].message.content.strip()

# Agenten-Funktionen: Jede Funktion loggt ihr Ergebnis, damit es per SSE an die Webseite gesendet wird.
def planner_agent(input_data):
    prompt = (
        "Du bist ein philosophischer Forschungsassistent. "
        "Extrahiere und liste in Stichpunkten offene und kontroverse philosophische Fragestellungen aus dem folgenden Text:\n\n" +
        input_data
    )
    messages = [
        {"role": "system", "content": "Du unterstützt bei der Identifikation philosophischer Fragen."},
        {"role": "user", "content": prompt}
    ]
    result = call_chat_completion(messages)
    log_message(f"Planner Agent: Fragestellungen extrahiert.\nErgebnis:\n{result}")
    questions = [line.strip("- ").strip() for line in result.splitlines() if line.strip()]
    return questions

def historical_agent(question):
    prompt = (
        f"Beschreibe klassische philosophische Ansätze und Theorien, die zu folgender Frage passen: {question}\n"
        "Fasse relevante historische Perspektiven prägnant zusammen."
    )
    messages = [
        {"role": "system", "content": "Du bist Experte für klassische Philosophie."},
        {"role": "user", "content": prompt}
    ]
    result = call_chat_completion(messages)
    log_message(f"Historical Agent: Klassische Ansätze extrahiert.\nErgebnis:\n{result}")
    return result

def modernity_agent(question):
    prompt = (
        f"Beschreibe zeitgenössische philosophische Diskurse und moderne Ansätze zur Frage: {question}\n"
        "Gib einen Überblick über aktuelle Debatten."
    )
    messages = [
        {"role": "system", "content": "Du bist Experte für moderne philosophische Diskurse."},
        {"role": "user", "content": prompt}
    ]
    result = call_chat_completion(messages)
    log_message(f"Modernity Agent: Moderne Diskurse extrahiert.\nErgebnis:\n{result}")
    return result

def comparison_agent(classical_info, modern_info):
    prompt = (
        "Vergleiche und kontrastiere die folgenden beiden Perspektiven: \n\n"
        "Klassische Ansätze:\n" + classical_info + "\n\n"
        "Zeitgenössische Ansätze:\n" + modern_info + "\n\n"
        "Erstelle eine Zusammenfassung, die Ähnlichkeiten, Unterschiede und kritische Spannungsfelder aufzeigt."
    )
    messages = [
        {"role": "system", "content": "Du führst einen Vergleich philosophischer Ansätze durch."},
        {"role": "user", "content": prompt}
    ]
    result = call_chat_completion(messages)
    log_message(f"Comparison Agent: Ansätze verglichen.\nErgebnis:\n{result}")
    return result

def synthesis_agent(comparison_info):
    prompt = (
        "Basierend auf der folgenden vergleichenden Analyse, erarbeite eine kohärente Synthese, "
        "die die zentralen Erkenntnisse in eine konsistente philosophische Argumentation einbettet:\n\n" +
        comparison_info
    )
    messages = [
        {"role": "system", "content": "Du bist Experte im Syntheseprozess philosophischer Argumente."},
        {"role": "user", "content": prompt}
    ]
    result = call_chat_completion(messages)
    log_message(f"Synthesis Agent: Synthese erstellt.\nErgebnis:\n{result}")
    return result

def evaluation_agent(synthesis):
    prompt = (
        "Bewerte die folgende Synthese hinsichtlich ihrer Kohärenz, Tiefe und Vollständigkeit. "
        "Antworte nur mit 'Yes' oder 'No', wobei 'Yes' bedeutet, dass die Synthese ausreichend ist:\n\n" +
        synthesis
    )
    messages = [
        {"role": "system", "content": "Du evaluierst philosophische Argumente kritisch."},
        {"role": "user", "content": prompt}
    ]
    result = call_chat_completion(messages)
    log_message(f"Evaluation Agent: Synthese bewertet.\nErgebnis:\n{result}")
    return "yes" in result.lower()

def detail_analysis_agent(question, current_synthesis):
    prompt = (
        "Verfeinere und erweitere die folgende Synthese, um eventuelle Lücken oder Schwächen zu schließen. "
        "Nutze dazu weitere Details zu der Frage: " + question + "\n\n"
        "Aktuelle Synthese:\n" + current_synthesis
    )
    messages = [
        {"role": "system", "content": "Du bist Experte in der Detailanalyse und Argumentationsverfeinerung."},
        {"role": "user", "content": prompt}
    ]
    result = call_chat_completion(messages)
    log_message(f"Detail Analysis Agent: Synthese verfeinert.\nErgebnis:\n{result}")
    return result

def report_aggregator(synthesis):
    prompt = (
        "Erstelle einen ausführlichen, gut lesbaren Bericht in reinem Textformat, "
        "der folgende Elemente enthält:\n"
        "- Eine kurze Zusammenfassung der Synthese\n"
        "- Eine ausführliche Darstellung der Argumentation\n"
        "- Hinweise auf verwendete Quellen oder Ansätze, falls vorhanden\n\n"
        "Nutze die folgende Synthese als Grundlage:\n" + synthesis
    )
    messages = [
        {"role": "system", "content": "Du generierst einen finalen, gut lesbaren Bericht in reinem Textformat."},
        {"role": "user", "content": prompt}
    ]
    result = call_chat_completion(messages, max_completion_tokens=300)
    log_message(f"Report Aggregator: Bericht generiert.\nErgebnis:\n{result}")
    return result

def feedback_agent(report):
    prompt = (
        "Bewerte den folgenden Bericht hinsichtlich Klarheit, Gründlichkeit und Qualität. "
        "Antworte nur mit 'Yes' oder 'No', wobei 'Yes' bedeutet, dass der Bericht zufriedenstellend ist:\n\n" +
        report
    )
    messages = [
        {"role": "system", "content": "Du bist ein kritischer Gutachter."},
        {"role": "user", "content": prompt}
    ]
    result = call_chat_completion(messages)
    log_message(f"Feedback Agent: Bericht bewertet.\nErgebnis:\n{result}")
    return "yes" in result.lower()

def check_time(start_time, max_time):
    """Überprüft die verbleibende Zeit und gibt Warnungen aus."""
    elapsed = time.time() - start_time
    remaining = max_time - elapsed
    if remaining <= 10:
        log_message(f"Achtung: Nur noch {int(remaining)} Sekunden verbleibend!")
    if elapsed >= max_time:
        log_message("Maximale Zeit erreicht, Prozess wird finalisiert.")
        return False
    return True

def run_agent_system(input_data, max_time):
    """Führt das Agentensystem mit Zeitlimit aus und speichert die finalen Ergebnisse."""
    global final_results, processing_done
    start_time = time.time()
    final_reports = []
    log_message("Agentensystem gestartet.")
    
    questions = planner_agent(input_data)
    log_message(f"{len(questions)} Fragestellungen identifiziert.")
    
    for question in questions:
        log_message(f"Beginne Bearbeitung der Frage: {question}")
        classical_info = historical_agent(question)
        if not check_time(start_time, max_time):
            break
        modern_info = modernity_agent(question)
        if not check_time(start_time, max_time):
            break
        comparison_info = comparison_agent(classical_info, modern_info)
        if not check_time(start_time, max_time):
            break
        synthesis = synthesis_agent(comparison_info)
        if not check_time(start_time, max_time):
            break

        attempt = 1
        while not evaluation_agent(synthesis):
            if not check_time(start_time, max_time):
                log_message("Zeitlimit erreicht während Detailanalyse, fahre mit aktueller Synthese fort.")
                break
            log_message(f"Verfeinerungsdurchlauf {attempt} für Frage: {question}")
            synthesis = detail_analysis_agent(question, synthesis)
            attempt += 1
            time.sleep(1)
        
        report = report_aggregator(synthesis)
        if not check_time(start_time, max_time):
            log_message("Zeitlimit erreicht vor Feedbackschleife, verwende aktuellen Bericht.")
        feedback_attempt = 1
        while not feedback_agent(report):
            if not check_time(start_time, max_time):
                log_message("Zeitlimit erreicht während Feedback, verwende aktuellen Bericht.")
                break
            log_message(f"Feedback-Durchlauf {feedback_attempt} für Frage: {question}")
            synthesis = detail_analysis_agent(question, synthesis)
            report = report_aggregator(synthesis)
            feedback_attempt += 1
            time.sleep(1)
        
        final_reports.append({"question": question, "report": report})
    
    final_results = final_reports
    processing_done = True
    log_message("Agentensystem abgeschlossen.")

# Hintergrundthread startet das Agentensystem
def background_process(input_data, max_time):
    run_agent_system(input_data, max_time)

@app.route("/", methods=["GET", "POST"])
def index():
    global processing_done, final_results, progress_queue
    if request.method == "POST":
        input_text = request.form.get("input_text")
        max_time = int(request.form.get("max_time", "120"))
        processing_done = False
        final_results = None
        while not progress_queue.empty():
            progress_queue.get()
        thread = threading.Thread(target=background_process, args=(input_text, max_time))
        thread.start()
        return render_template("index.html", input_text=input_text)
    return render_template("index.html", input_text="")

# SSE-Route für Live-Progress
@app.route("/progress")
def progress():
    def event_stream():
        while not processing_done or not progress_queue.empty():
            try:
                msg = progress_queue.get(timeout=1)
                yield f"data: {msg}\n\n"
            except Exception:
                continue
    return Response(event_stream(), mimetype="text/event-stream")

# Route, um finale Ergebnisse als HTML abzurufen
@app.route("/results")
def results():
    if processing_done and final_results is not None:
        html_result = ""
        for item in final_results:
            html_result += f"<div class='result'><h3>Frage: {item['question']}</h3><pre>{item['report']}</pre></div>"
        return html_result
    else:
        return ""

if __name__ == "__main__":
    app.run(debug=True)
