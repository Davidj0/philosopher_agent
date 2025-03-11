import openai
from flask import Flask, render_template, request, Response, jsonify
import threading, queue, time
import os
from dotenv import load_dotenv
import openai

# .env-Datei laden
load_dotenv()

# API-Key aus der Umgebung laden
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

# Globale Variablen für den Fortschritts-Log und die finalen Ergebnisse
progress_queue = queue.Queue()
final_results = None
processing_done = False

def log_message(message):
    """Schreibt eine Nachricht in die globale Fortschritts-Queue."""
    print(message)  # Für die Konsole (optional)
    progress_queue.put(message)

def call_chat_completion(messages, model="gpt-4o-mini", temperature=0.7, max_completion_tokens=150):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens
    )
    return response.choices[0].message.content.strip()

# Agenten-Funktionen (wie zuvor definiert)
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
    log_message("Planner Agent: Fragestellungen extrahiert.")
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
    log_message("Historical Agent: Klassische Ansätze extrahiert.")
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
    log_message("Modernity Agent: Moderne Diskurse extrahiert.")
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
    log_message("Comparison Agent: Ansätze verglichen.")
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
    log_message("Synthesis Agent: Synthese erstellt.")
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
    log_message("Evaluation Agent: Synthese bewertet.")
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
    log_message("Detail Analysis Agent: Synthese verfeinert.")
    return result

def report_aggregator(synthesis):
    prompt = (
        "Erstelle einen strukturierten Bericht in JSON-Format, der die folgenden Elemente enthält:\n"
        "- summary: Eine kurze Zusammenfassung der Synthese\n"
        "- detailed_argument: Die ausführliche Darstellung des Arguments\n"
        "- references: Hinweise auf verwendete Quellen oder Ansätze\n\n"
        "Nutze die folgende Synthese als Grundlage:\n" + synthesis
    )
    messages = [
        {"role": "system", "content": "Du generierst einen strukturierten, finalen Bericht."},
        {"role": "user", "content": prompt}
    ]
    result = call_chat_completion(messages)
    log_message("Report Aggregator: Bericht generiert.")
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
    log_message("Feedback Agent: Bericht bewertet.")
    return "yes" in result.lower()

def check_time(start_time, max_time):
    """Überprüft, wie viel Zeit noch übrig ist, und gibt Warnungen aus."""
    elapsed = time.time() - start_time
    remaining = max_time - elapsed
    if remaining <= 10:
        log_message(f"Achtung: Nur noch {int(remaining)} Sekunden verbleibend!")
    if elapsed >= max_time:
        log_message("Maximale Zeit erreicht, Prozess wird finalisiert.")
        return False
    return True

def run_agent_system(input_data, max_time):
    """Führt das Agentensystem aus, unter Berücksichtigung eines Zeitlimits."""
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
        # Evaluationsschleife: Wenn Synthese unzureichend ist oder Zeit fast abgelaufen ist,
        # dann wird verfeinert – falls die Zeit überschritten wird, wird abgebrochen.
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
            log_message("Zeitlimit erreicht vor Feedbackschleife, fahre mit aktuellem Bericht fort.")
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

# Hintergrundthread-Funktion, die das Agentensystem ausführt
def background_process(input_data, max_time):
    run_agent_system(input_data, max_time)

# Route für die Hauptseite
@app.route("/", methods=["GET", "POST"])
def index():
    global processing_done, final_results, progress_queue
    if request.method == "POST":
        input_text = request.form.get("input_text")
        max_time = int(request.form.get("max_time", "120"))  # Zeitlimit in Sekunden, Standard: 120 Sekunden
        # Setze globale Variablen zurück
        processing_done = False
        final_results = None
        while not progress_queue.empty():
            progress_queue.get()
        # Starte das Agentensystem in einem Hintergrundthread
        thread = threading.Thread(target=background_process, args=(input_text, max_time))
        thread.start()
        return render_template("index.html", results=None, input_text=input_text)
    return render_template("index.html", results=None)

# SSE-Route für den Live-Progress
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

# Route, um finale Ergebnisse abzurufen
@app.route("/results")
def results():
    if processing_done and final_results is not None:
        return jsonify({"results": final_results})
    else:
        return jsonify({"results": []})

if __name__ == "__main__":
    app.run(debug=True)