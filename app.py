import openai
from flask import Flask, render_template, request
import time
import os
from dotenv import load_dotenv
import openai

# .env-Datei laden
load_dotenv()

# API-Key aus der Umgebung laden
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

def call_chat_completion(messages, model="gpt-4o-mini", temperature=0.7, max_completion_tokens=150):
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens
    )
    return response.choices[0].message.content.strip()

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
    return call_chat_completion(messages)

def modernity_agent(question):
    prompt = (
        f"Beschreibe zeitgenössische philosophische Diskurse und moderne Ansätze zur Frage: {question}\n"
        "Gib einen Überblick über aktuelle Debatten."
    )
    messages = [
        {"role": "system", "content": "Du bist Experte für moderne philosophische Diskurse."},
        {"role": "user", "content": prompt}
    ]
    return call_chat_completion(messages)

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
    return call_chat_completion(messages)

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
    return call_chat_completion(messages)

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
    return call_chat_completion(messages)

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
    return call_chat_completion(messages)

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
    return "yes" in result.lower()

def run_agent_system(input_data):
    questions = planner_agent(input_data)
    final_reports = []
    
    for question in questions:
        classical_info = historical_agent(question)
        modern_info = modernity_agent(question)
        comparison_info = comparison_agent(classical_info, modern_info)
        synthesis = synthesis_agent(comparison_info)
        
        attempt = 1
        while not evaluation_agent(synthesis):
            synthesis = detail_analysis_agent(question, synthesis)
            attempt += 1
            time.sleep(1)
        
        report = report_aggregator(synthesis)
        feedback_attempt = 1
        while not feedback_agent(report):
            synthesis = detail_analysis_agent(question, synthesis)
            report = report_aggregator(synthesis)
            feedback_attempt += 1
            time.sleep(1)
        
        final_reports.append({"question": question, "report": report})
    return final_reports

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        input_text = request.form.get("input_text")
        results = run_agent_system(input_text)
        return render_template("index.html", results=results, input_text=input_text)
    return render_template("index.html", results=None)

if __name__ == "__main__":
    app.run(debug=True)
