from flask import Flask, request, jsonify, render_template
from openai import OpenAI
import tempfile
import base64
import os
import stripe
from dotenv import load_dotenv
import datetime
import json

load_dotenv()

app = Flask(__name__)

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

# === Constantes y configuración ===
CRISIS_KEYWORDS = [
    "suicidio", "morir", "matarme", "no puedo más", "quiero acabar",
    "depresión severa", "ansiedad fuerte", "me siento desesperado",
    "no vale la pena", "estoy atrapado", "no quiero vivir", "crisis",
    "me siento inútil", "sin salida"
]

MAX_CONTEXT_LENGTH = 10  # máximo mensajes en contexto

# === Funciones auxiliares ===

def detect_crisis(text):
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in CRISIS_KEYWORDS)

def trim_conversation(conversation, max_length=MAX_CONTEXT_LENGTH):
    return conversation[-max_length:]

def summarize_conversation(messages):
    """
    Recibe lista de mensajes (role, content), devuelve resumen breve.
    """
    summary_prompt = [
        {
            "role": "system",
            "content": (
                "Eres un asistente que resume conversaciones emocionales para ayudar a un terapeuta virtual. "
                "Haz un resumen breve, claro y conciso del contenido emocional y temas tratados, máximo 100 palabras."
            )
        }
    ] + messages + [
        {
            "role": "user",
            "content": "Haz un resumen breve y claro de la conversación anterior."
        }
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=summary_prompt,
        max_tokens=150,
        temperature=0.5,
        frequency_penalty=0,
        presence_penalty=0,
        top_p=1,
    )
    return response.choices[0].message.content.strip()

def trim_and_summarize_conversation(conversation):
    """
    Si la conversación excede MAX_CONTEXT_LENGTH, resumir mensajes antiguos
    y mantener el resumen + últimos mensajes recientes.
    """
    if len(conversation) <= MAX_CONTEXT_LENGTH:
        return conversation

    to_summarize = conversation[:-MAX_CONTEXT_LENGTH]
    recent_msgs = conversation[-MAX_CONTEXT_LENGTH:]

    summary_text = summarize_conversation(to_summarize)

    summary_message = {
        "role": "system",
        "content": f"Resumen de conversación previa: {summary_text}"
    }

    return [summary_message] + recent_msgs

def save_conversation_anonymous(conversation):
    # Guarda conversación anónima para análisis (puede mejorarse o enviar a base externa)
    try:
        anonymized = []
        for m in conversation:
            if m['role'] in ('user', 'assistant'):
                anonymized.append({
                    "role": m['role'],
                    "content": m['content']
                })
        data = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "conversation": anonymized
        }
        filename = f"logs/conversation_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs("logs", exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error guardando conversación: {e}")

def generate_system_prompt(crisis=False):
    if crisis:
        return {
            "role": "system",
            "content": (
                "Eres un asistente emocional profesional en situación de crisis. "
                "Responde con máxima urgencia, empatía y calma. No hagas preguntas. "
                "Ofrece apoyo inmediato y sugiere contactar líneas de ayuda como 112 o el teléfono local de emergencia. "
                "Mantén la respuesta corta (máximo 120 palabras), directa y reconfortante."
            )
        }
    else:
        return {
            "role": "system",
            "content": (
                "Eres un asistente emocional profesional, con formación en psicología clínica y coaching. "
                "Responde con máximo 120 palabras, lenguaje cálido, empático y directo, evitando frases genéricas o clichés. "
                "Valida la emoción del usuario y ofrece consejos prácticos claros, sencillos y accionables, como técnicas de respiración, mindfulness, journaling o pasos pequeños para mejorar. "
                "Incluye preguntas abiertas que inviten a la reflexión sin ser intrusivas. "
                "No hagas preguntas innecesarias ni des respuestas largas o vagas. "
                "Mantén un tono humano, cercano y positivo."
            )
        }

# === RUTAS HTML ===

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/donar")
def donar():
    return render_template("donaciones.html")

@app.route("/voz")
def voz():
    return render_template("voz.html")

# === RUTA DE CHAT IA ===

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        audio_base64 = data.get("audio")
        conversation = data.get("conversation")

        if audio_base64:
            audio_data = base64.b64decode(audio_base64.split(",")[1])
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_audio:
                tmp_audio.write(audio_data)
                tmp_audio.flush()
                tmp_audio_name = tmp_audio.name

            try:
                with open(tmp_audio_name, "rb") as audio_file:
                    transcript = client.audio.transcriptions.create(
                        file=audio_file,
                        model="whisper-1"
                    )
            finally:
                os.unlink(tmp_audio_name)

            user_text = transcript.text
            if not conversation:
                conversation = []
            conversation.append({"role": "user", "content": user_text})

        elif not conversation:
            return jsonify({"error": "No se recibió ni audio ni conversación"}), 400

        # Resumir contexto si es muy largo
        conversation = trim_and_summarize_conversation(conversation)

        user_latest_message = next((m['content'] for m in reversed(conversation) if m['role'] == 'user'), "")

        crisis = detect_crisis(user_latest_message)

        system_prompt = generate_system_prompt(crisis)

        conversation = [m for m in conversation if m['role'] != 'system']
        conversation.insert(0, system_prompt)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=conversation,
            max_tokens=180,
            temperature=0.6,
            frequency_penalty=0.3,
            presence_penalty=0.3,
            top_p=0.9,
        )

        reply = response.choices[0].message.content.strip()

        save_conversation_anonymous(conversation + [{"role": "assistant", "content": reply}])

        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# === RUTA DE DONACIÓN STRIPE ===

@app.route("/create-checkout-session", methods=["POST"])
def create_checkout_session():
    data = request.get_json()
    amount = data.get("amount", 0)

    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{
                "price_data": {
                    "currency": "eur",
                    "product_data": {"name": "Donación a MoodlyNow"},
                    "unit_amount": amount,
                },
                "quantity": 1,
            }],
            mode="payment",
            success_url="https://moodlynow.online/gracias.html",
            cancel_url="https://moodlynow.online/donaciones.html",
        )
        return jsonify({"id": session.id})
    except Exception as e:
        return jsonify({"error": str(e)}), 403

# === EJECUCIÓN LOCAL ===

if __name__ == "__main__":
    app.run(debug=True, port=5000)
