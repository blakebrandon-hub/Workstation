"""
Workstation + Assistant — Flask server
"""

import os
import base64
import re
import time
import html as html_module
import logging
import requests
from typing import List, Dict

from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
from google import genai
from google.genai import types

from engine import Workspace
from assistant import ASSISTANT_SYSTEM_PROMPT, parse_tags, strip_tags, route_tags, AssistantMemory

logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ── Workstation (sandbox + manual memory/functions) ──
ws = Workspace()

# ── Assistant (AI + Supabase tables) ──
GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
gemini_key = os.environ.get("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=gemini_key)
memory = AssistantMemory()


# ─────────────────────────────────────────────────────────────────────────────
# ASSISTANT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def build_system_prompt():
    mem_ctx = memory.memory_context()
    fn_names = list(memory.get_functions().keys())
    fn_ctx = f"\nASSISTANT'S FUNCTIONS: {', '.join(fn_names)}" if fn_names else ""
    return ASSISTANT_SYSTEM_PROMPT + ("\n\n" + mem_ctx if mem_ctx else "") + fn_ctx


SUMMARY_SYSTEM = (
    "You are a concise archivist. Given a block of conversation, write a dense, "
    "factual 3-5 sentence summary capturing the key topics, decisions, and context. "
    "Write in third person (e.g. 'The user asked...', 'The assistant explained...'). "
    "Do not add opinions or padding."
)

def _embed_text(text: str) -> list | None:
    """Generate an embedding via Gemini. Returns None on failure."""
    try:
        result = gemini_client.models.embed_content(
            model="text-embedding-004",
            contents=[text],
        )
        return result.embeddings[0].values
    except Exception as e:
        logger.warning(f"Embedding failed: {e}")
        return None

def maybe_summarise():
    """Check whether a new summary batch is ready; if so, summarise and embed it."""
    batch = memory.pending_summary_batch()
    if not batch:
        return

    # Build a transcript of the batch
    transcript = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in batch
    )
    try:
        summary_text = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=transcript,
            config=types.GenerateContentConfig(
                system_instruction=SUMMARY_SYSTEM,
                temperature=0.3,
            )
        ).text.strip()
    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        return

    covers_ids = [m["id"] for m in batch]
    summary_id = memory.write_summary(summary_text, covers_ids)

    # Embed and push to cascade
    embedding = _embed_text(summary_text)
    if embedding and summary_id:
        memory.update_summary_embedding(summary_id, embedding)

    logger.info(f"Summary written (id={summary_id}) covering message ids {covers_ids}")

def build_gemini_contents(history: list[dict], current_file_uri: str = None,
                           current_mime_type: str = None, current_image_b64: str = None) -> list:
    contents = []
    for i, msg in enumerate(history):
        role = "user" if msg["role"] == "user" else "model"
        text_content = msg["content"]
        if not text_content and i == len(history) - 1 and (current_file_uri or current_image_b64):
            text_content = "[file attached]"
        parts = [types.Part(text=text_content)]
        if i == len(history) - 1 and role == "user":
            if current_file_uri and current_mime_type:
                parts.append(types.Part(
                    file_data=types.FileData(file_uri=current_file_uri, mime_type=current_mime_type)
                ))
            elif current_image_b64:
                parts.append(types.Part(
                    inline_data=types.Blob(
                        mime_type="image/png",
                        data=base64.b64decode(current_image_b64)
                    )
                ))
        contents.append(types.Content(role=role, parts=parts))
    return contents

def call_assistant(user_message: str, file_uri: str = None, mime_type: str = None,
               image_b64: str = None) -> dict:
    history = memory.load_conversation()
    history.append({"role": "user", "content": user_message})

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=build_gemini_contents(history, file_uri, mime_type, image_b64),
        config=types.GenerateContentConfig(
            system_instruction=build_system_prompt(),
            temperature=0.9,
        )
    )

    raw = response.text
    memory.append_message("user", user_message)
    user_msg_id = memory.get_last_message_id()
    memory.append_message("assistant", raw)
    assistant_msg_id = memory.get_last_message_id()

    tags = parse_tags(raw)
    prose = strip_tags(raw)
    events = route_tags(tags, memory)

    action_results = []
    for ev in events:
        if ev['type'] == 'execute':
            action_results.append(f"Code execution result:\n{ev.get('result')}")

    if action_results:
        observation = "[SYSTEM OBSERVATION]\n" + "\n".join(action_results)
        memory.append_message("user", observation)

    maybe_summarise()

    return {
        'prose': prose,
        'events': events,
        'raw': raw,
        'user_message_id': user_msg_id,
        'assistant_message_id': assistant_msg_id,
    }


# ─────────────────────────────────────────────────────────────────────────────
# WORKSTATION — MEMORY
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/memory', methods=['GET'])
def memory_all():
    return jsonify(ws.all_memories())

@app.route('/api/memory', methods=['POST'])
def memory_set():
    data = request.json
    key = data.get('key', '').strip()
    value = data.get('value', '').strip()
    if not key or not value:
        return jsonify({'error': 'key and value required'}), 400
    ws.remember(key, value)
    return jsonify({'ok': True, 'key': key, 'value': value})

@app.route('/api/memory/<key>', methods=['GET'])
def memory_get(key):
    value = ws.recall(key)
    if value is None:
        return jsonify({'error': 'not found'}), 404
    return jsonify({'key': key, 'value': value})

@app.route('/api/memory/<key>', methods=['DELETE'])
def memory_delete(key):
    ws.forget(key)
    return jsonify({'ok': True})


# ─────────────────────────────────────────────────────────────────────────────
# WORKSTATION — FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/functions', methods=['GET'])
def functions_all():
    return jsonify(ws.get_functions())

@app.route('/api/functions', methods=['POST'])
def function_write():
    data = request.json
    name = data.get('name', '').strip()
    code = data.get('code', '').strip()
    if not name or not code:
        return jsonify({'error': 'name and code required'}), 400
    ws.write_function(name, code)
    return jsonify({'ok': True, 'name': name})

@app.route('/api/functions/<name>', methods=['DELETE'])
def function_delete(name):
    ws.delete_function(name)
    return jsonify({'ok': True})


# ─────────────────────────────────────────────────────────────────────────────
# WORKSTATION — EXECUTE
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/run', methods=['POST'])
def run_code():
    data = request.json
    code = data.get('code', '').strip()
    if not code:
        return jsonify({'error': 'code required'}), 400
    result = ws.run(code)
    return jsonify({'ok': True, 'result': result})

@app.route('/api/invoke', methods=['POST'])
def invoke_fn():
    data = request.json
    call = data.get('call', '').strip()
    if not call:
        return jsonify({'error': 'call required'}), 400
    result = ws.invoke(call)
    return jsonify({'ok': True, 'result': result})


# ─────────────────────────────────────────────────────────────────────────────
# WORKSTATION — STATE
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/state', methods=['GET'])
def state():
    return jsonify({
        'memories': ws.all_memories(),
        'functions': ws.get_functions(),
        'log': ws.load_log(50)
    })

@app.route('/api/log/clear', methods=['POST'])
def clear_log():
    ws.clear_log()
    return jsonify({'ok': True})


# ─────────────────────────────────────────────────────────────────────────────
# ASSISTANT — CHAT
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/chat', methods=['POST'])
def chat():
    message = request.form.get('message', '').strip()
    file = request.files.get('file')
    image_b64 = request.form.get('image_b64')

    file_uri = None
    mime_type = None

    if file:
        temp_path = f"temp_{file.filename}"
        file.save(temp_path)
        uploaded_file = gemini_client.files.upload(file=temp_path)
        file_uri = uploaded_file.uri
        mime_type = file.content_type
        os.remove(temp_path)
        result = call_assistant(message, file_uri, mime_type)
    elif image_b64:
        result = call_assistant(message, image_b64=image_b64)
    else:
        result = call_assistant(message)

    return jsonify(result)


# ─────────────────────────────────────────────────────────────────────────────
# ASSISTANT — MEMORY / EMBEDDING / CASCADE
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/assistant/state', methods=['GET'])
def assistant_state():
    return jsonify({
        'memories': memory.all_memories(),
        'functions': memory.get_functions(),
        'summaries': memory.get_summaries(),
        'history_length': len(memory.load_conversation())
    })

@app.route('/api/context', methods=['GET'])
def context():
    history = memory.load_conversation()
    system_prompt = ASSISTANT_SYSTEM_PROMPT
    gemini_contents = build_gemini_contents(history)
    return jsonify({
        'system_prompt': system_prompt,
        'conversation_history': history,
        'gemini_contents': [
            {
                'role': c.role,
                'parts': [
                    {'text': p.text} if hasattr(p, 'text') else
                    {'file_uri': p.file_data.file_uri} if hasattr(p, 'file_data') else str(p)
                    for p in c.parts
                ]
            }
            for c in gemini_contents
        ],
        'total_messages': len(history),
        'memory_entries': memory.all_memories(),
        'summaries': memory.summary_context(),
        'functions': list(memory.get_functions().keys())
    })

@app.route('/api/cascade', methods=['POST'])
def cascade():
    data = request.json
    embedding = data.get('embedding')
    limit = data.get('limit', 6)
    threshold = data.get('threshold', 0.3)
    if not embedding:
        return jsonify({'error': 'embedding required'}), 400
    results = memory.cascade_search(embedding, threshold, limit)
    return jsonify({'results': results})

@app.route('/api/embed', methods=['POST'])
def embed():
    data = request.json
    table = data.get('table')
    key = data.get('key')
    embedding = data.get('embedding')
    if table == 'memory':
        memory.update_memory_embedding(key, embedding)
    elif table == 'conversation':
        memory.update_conversation_embedding(key, embedding)
    return jsonify({'success': True})

@app.route('/api/conversation', methods=['GET'])
def conversation():
    res = memory._db.table("assistant_conversation").select("id,role,content").order("id").execute()
    return jsonify([{"id": r["id"], "role": r["role"], "content": r["content"]} for r in res.data])



# ─────────────────────────────────────────────────────────────────────────────
# SHARED
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model': GEMINI_MODEL})

@app.route('/')
def index():
    return render_template('index.html')


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7000))
    print(f"\n✦ Workstation running on http://localhost:{port}\n")
    app.run(debug=False, port=port)