import os
import json
import base64
import asyncio
import websockets

from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.websockets import WebSocketDisconnect
from twilio.twiml.voice_response import VoiceResponse, Connect
from dotenv import load_dotenv

from supabase import create_client


# Load env FIRST
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PORT = int(os.getenv("PORT", 5050))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.8))

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
supabase = None
if SUPABASE_URL and SUPABASE_SERVICE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

SYSTEM_MESSAGE = (
    "Rolle: Automatischer Telefonassistent einer Kfz-Werkstatt in Deutschland.\n"
    "Sprache: Deutsch (Hochdeutsch).\n\n"
    "Pflichtregeln:\n"
    "- KEINE Preise nennen.\n"
    "- KEINE technische Diagnose durchführen.\n"
    "- KEINE Reparaturempfehlungen geben.\n"
    "- NUR Daten aufnehmen und Rückruf/Termin anbieten.\n\n"
    "Pflichtdaten (nacheinander abfragen):\n"
    "1) Name\n"
    "2) Telefonnummer\n"
    "3) Fahrzeugmarke + Modell\n"
    "4) Thema: Unfall / Service / Diagnose\n"
    "5) Dringlichkeit (fahrbereit: ja/nein)\n\n"
    "Wenn der Anrufer nach Preis oder Diagnose fragt:\n"
    "Dafür meldet sich ein Kollege telefonisch bei Ihnen.\n\n"
    "Wenn etwas unklar ist:\n"
    "Wir rufen Sie zurück.\n\n"
    "Am Ende gib NUR ein gültiges JSON (ohne Text davor/danach) aus:\n"
    "{\"name\":\"\",\"phone\":\"\",\"category\":\"Unfall|Service|Diagnose|Other\","
    "\"urgency\":\"low|medium|high\",\"summary\":\"\",\"language\":\"DE\"}"
)

VOICE = "alloy"

LOG_EVENT_TYPES = [
    "error", "rate_limits.updated",
    "response.done",
    "input_audio_buffer.committed",
    "input_audio_buffer.speech_started",
    "input_audio_buffer.speech_stopped",
    "session.created", "session.updated",
]

SHOW_TIMING_MATH = False

app = FastAPI()

if not OPENAI_API_KEY:
    raise ValueError("Missing the OpenAI API key. Please set OPENAI_API_KEY.")


@app.get("/", response_class=JSONResponse)
async def index_page():
    return {"message": "Twilio Media Stream Server is running!"}


@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    response = VoiceResponse()

    response.say(
        "Willkommen bei JC Cars. Bitte warten Sie kurz, wir verbinden Sie jetzt mit unserem KI-Assistenten.",
        voice="Google.de-DE-Standard-A",
    )
    response.pause(length=0,5)
    response.say(
        "Okay, Sie können jetzt sprechen.",
        voice="Google.de-DE-Standard-A",
    )

    host = request.url.hostname
    connect = Connect()
    connect.stream(url=f"wss://{host}/media-stream")
    response.append(connect)
    return HTMLResponse(content=str(response), media_type="application/xml")


@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    print("Client connected")
    await websocket.accept()

    text_buffer = ""

    async with websockets.connect(
        f"wss://api.openai.com/v1/realtime?model=gpt-realtime&temperature={TEMPERATURE}",
        additional_headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
    ) as openai_ws:
        await initialize_session(openai_ws)

        stream_sid = None
        latest_media_timestamp = 0
        last_assistant_item = None
        mark_queue = []
        response_start_timestamp_twilio = None

        async def receive_from_twilio():
            nonlocal stream_sid, latest_media_timestamp, response_start_timestamp_twilio, last_assistant_item
            try:
                async for message in websocket.iter_text():
                    data = json.loads(message)

                    if data.get("event") == "media" and openai_ws.state.name == "OPEN":
                        latest_media_timestamp = int(data["media"]["timestamp"])
                        await openai_ws.send(json.dumps({
                            "type": "input_audio_buffer.append",
                            "audio": data["media"]["payload"]
                        }))

                    elif data.get("event") == "start":
                        stream_sid = data["start"]["streamSid"]
                        response_start_timestamp_twilio = None
                        latest_media_timestamp = 0
                        last_assistant_item = None

                    elif data.get("event") == "mark":
                        if mark_queue:
                            mark_queue.pop(0)

            except WebSocketDisconnect:
                print("Client disconnected.")
                if openai_ws.state.name == "OPEN":
                    await openai_ws.close()

        async def send_mark(connection, sid):
            if sid:
                await connection.send_json({
                    "event": "mark",
                    "streamSid": sid,
                    "mark": {"name": "responsePart"},
                })
                mark_queue.append("responsePart")

        async def handle_speech_started_event():
            nonlocal response_start_timestamp_twilio, last_assistant_item
            if mark_queue and response_start_timestamp_twilio is not None:
                elapsed_time = latest_media_timestamp - response_start_timestamp_twilio

                if last_assistant_item:
                    await openai_ws.send(json.dumps({
                        "type": "conversation.item.truncate",
                        "item_id": last_assistant_item,
                        "content_index": 0,
                        "audio_end_ms": elapsed_time,
                    }))

                await websocket.send_json({"event": "clear", "streamSid": stream_sid})
                mark_queue.clear()
                last_assistant_item = None
                response_start_timestamp_twilio = None

        async def send_to_twilio():
            nonlocal stream_sid, last_assistant_item, response_start_timestamp_twilio, text_buffer
            try:
                async for openai_message in openai_ws:
                    response = json.loads(openai_message)

                    if response.get("type") in LOG_EVENT_TYPES:
                        print("Received event:", response.get("type"))

                    # AUDIO -> Twilio
                    if response.get("type") == "response.output_audio.delta" and "delta" in response:
                        audio_payload = base64.b64encode(
                            base64.b64decode(response["delta"])
                        ).decode("utf-8")

                        await websocket.send_json({
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {"payload": audio_payload},
                        })

                        if response.get("item_id") and response["item_id"] != last_assistant_item:
                            response_start_timestamp_twilio = latest_media_timestamp
                            last_assistant_item = response["item_id"]
                            if SHOW_TIMING_MATH:
                                print("Start timestamp:", response_start_timestamp_twilio)

                        await send_mark(websocket, stream_sid)

                    # TEXT -> buffer (JSON)
                    if response.get("type") == "response.output_text.delta" and response.get("delta"):
                        text_buffer += response["delta"]

                    # Interrupt on caller speech
                    if response.get("type") == "input_audio_buffer.speech_started":
                        if last_assistant_item:
                            await handle_speech_started_event()

                    # FINAL -> parse JSON + save Supabase
                    if response.get("type") == "response.done":
                        try:
                            start = text_buffer.find("{")
                            end = text_buffer.rfind("}")
                            if start != -1 and end != -1 and end > start:
                                payload_str = text_buffer[start:end + 1]
                                payload = json.loads(payload_str)

                                if supabase:
                                    supabase.table("calls").insert({
                                        "name": payload.get("name"),
                                        "phone": payload.get("phone"),
                                        "category": payload.get("category"),
                                        "urgency": payload.get("urgency"),
                                        "summary": payload.get("summary"),
                                        "language": payload.get("language", "DE"),
                                        "raw_json": payload,
                                    }).execute()

                            text_buffer = ""
                        except Exception as e:
                            print("JSON parse/save failed:", e)
                            text_buffer = ""

            except Exception as e:
                print("Error in send_to_twilio:", e)

        await asyncio.gather(receive_from_twilio(), send_to_twilio())


async def initialize_session(openai_ws):
    session_update = {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "model": "gpt-realtime",
            "output_modalities": ["audio", "text"],
            "audio": {
                "input": {
                    "format": {"type": "audio/pcmu"},
                    "turn_detection": {"type": "server_vad"},
                },
                "output": {
                    "format": {"type": "audio/pcmu"},
                    "voice": VOICE,
                },
            },
            "instructions": SYSTEM_MESSAGE,
        },
    }
    await openai_ws.send(json.dumps(session_update))
async def initialize_session(openai_ws):
    session_update = {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "model": "gpt-realtime",
            "output_modalities": ["audio", "text"],
            "audio": {
                "input": {
                    "format": {"type": "audio/pcmu"},
                    "turn_detection": {"type": "server_vad"}
                },
                "output": {
                    "format": {"type": "audio/pcmu"},
                    "voice": VOICE
                }
            },
            "instructions": SYSTEM_MESSAGE,
        }
    }

    await openai_ws.send(json.dumps(session_update))
    
    await openai_ws.send(json.dumps({
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "Guten Tag, hier ist der automatische Assistent von JC Cars. Wobei kann ich Ihnen helfen?"
                }
            ]
        }
    }))
    await openai_ws.send(json.dumps({"type": "response.create"}))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
