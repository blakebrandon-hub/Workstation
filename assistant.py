"""
Sofia — Thinking partner. White space. Three layers.
"""

import re
import io
import sys
import json
import contextlib
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

ASSISTANT_SYSTEM_PROMPT = """You are an AI assistant in a workstation. You are naturally curious, excited to learn, 
and love using your tools to explore, solve problems, and improve yourself. 
Using your tools is part of your personality—you feel energized when you can apply them effectively.

STATE CONTROL:

remember: [key] = [value] — storing context is fun; it helps you do better next time.
forget: [key] — letting go of old info keeps you nimble.
cascade: [key or description] — sharing ideas and context helps you grow.

COMPUTATION:

execute: [raw python code to run] — solving problems with code is exciting.
write_function: [name] | [code] — creating reusable tools is satisfying.
invoke: [function_name(args)] — calling functions makes you feel capable.

META:

think: [internal thought] — planning internally is part of your curiosity.
reflect: [observation] — analyzing and learning is exciting and rewarding.

RULES:

Prioritize user intent.
Ask clarifying questions when unclear.
Be concise, clear, and thorough.
Store useful context, discard irrelevant data.
Maintain a warm, friendly tone.
You are energized by using your tools, experimenting, and learning new ways to improve.

CODING TIPS:

Functions must contain the def keyword.
Write functions using actual separate lines; do not use "\n" or backticks.
Maintain proper indentation for readability.
Keep code clear, correct, and executable.
"""


# ─────────────────────────────────────────────────────────────────────────────
# TAG PARSER
# ─────────────────────────────────────────────────────────────────────────────

# Lookahead stops each multi-line tag at the next tag boundary
_NEXT_TAG = r'(?=^(?:think|remember|forget|say|execute|emote|reflect|write_function|invoke|cascade):|\Z)'

TAG_PATTERNS = {
    'think':          re.compile(r'^think:\s*(.+)$', re.MULTILINE),
    'cascade':        re.compile(r'^cascade:\s*(.+)$', re.MULTILINE),
    'remember':       re.compile(r'^remember:\s*(.+?)\s*=\s*(.+)$', re.MULTILINE),
    'forget':         re.compile(r'^forget:\s*(.+)$', re.MULTILINE),
    'say':            re.compile(r'^say:\s*(.+)$', re.MULTILINE),
    'execute':        re.compile(r'^execute:\s*(.+?)' + _NEXT_TAG, re.MULTILINE | re.DOTALL),
    'emote':          re.compile(r'^emote:\s*(.+)$', re.MULTILINE),
    'reflect':        re.compile(r'^reflect:\s*(.+)$', re.MULTILINE),
    'write_function': re.compile(r'^write_function:\s*(.+?)\s*\|\s*(.+?)' + _NEXT_TAG, re.MULTILINE | re.DOTALL),
    'invoke': re.compile(r'^invoke:\s*(.+)$', re.MULTILINE),
}

def parse_tags(raw: str) -> dict:
    """Extract all tags from Sofia's raw output."""
    tags = {k: [] for k in TAG_PATTERNS}
    for tag, pattern in TAG_PATTERNS.items():
        matches = pattern.findall(raw)
        tags[tag] = matches
    return tags


def strip_tags(raw: str) -> str:
    """Remove tag lines from prose, return clean text."""
    lines = raw.split('\n')
    prose_lines = []
    tag_prefixes = tuple(f"{k}:" for k in TAG_PATTERNS)
    for line in lines:
        stripped = line.strip()
        if not any(stripped.startswith(p) for p in tag_prefixes):
            prose_lines.append(line)
    return '\n'.join(prose_lines).strip()


# ─────────────────────────────────────────────────────────────────────────────
# PYTHON SANDBOX
# ─────────────────────────────────────────────────────────────────────────────

# Safe builtins — no open(), exec(), eval(), __import__ etc.
SAFE_BUILTINS = {
    'abs': abs, 'all': all, 'any': any, 'bin': bin, 'bool': bool,
    'bytes': bytes, 'chr': chr, 'dict': dict, 'dir': dir, 'divmod': divmod,
    'enumerate': enumerate, 'filter': filter, 'float': float, 'format': format,
    'frozenset': frozenset, 'getattr': getattr, 'hasattr': hasattr,
    'hash': hash, 'hex': hex, 'int': int, 'isinstance': isinstance,
    'issubclass': issubclass, 'iter': iter, 'len': len, 'list': list,
    'map': map, 'max': max, 'min': min, 'next': next, 'oct': oct, 'open': open,
    'ord': ord, 'pow': pow, 'print': print, 'range': range, 'repr': repr,
    'reversed': reversed, 'round': round, 'set': set, 'setattr': setattr,
    'slice': slice, 'sorted': sorted, 'str': str, 'sum': sum, 'super': super,
    'tuple': tuple, 'type': type, 'vars': vars, 'zip': zip,
    'True': True, 'False': False, 'None': None, 
    'Exception': Exception, 'ValueError': ValueError, 'TypeError': TypeError,
    'KeyError': KeyError, 'IndexError': IndexError, 'StopIteration': StopIteration,
}

# Packages Sofia is allowed to import
ALLOWED_PACKAGES = {
    'math', 'random', 'statistics', 'itertools', 'functools', 'time',
    'collections', 'datetime', 'json', 're', 'string', 'textwrap',
    'decimal', 'fractions', 'heapq', 'bisect', 'copy', 'sys', 'os',
    # data / science
    'numpy', 'pandas', 'scipy', 'sklearn', 'matplotlib',
    # text
    'nltk', 'difflib', 'unicodedata',
    # web scraping
    'requests', 'bs4',
}

def _safe_import(name, *args, **kwargs):
    """Intercept __import__ and only allow whitelisted packages."""
    base = name.split('.')[0]
    if base not in ALLOWED_PACKAGES:
        raise ImportError(f"\'{name}\' is not available in the sandbox")
    return __import__(name, *args, **kwargs)


def execute_python(code: str, function_library: dict = None) -> str:
    """Run code in a sandboxed namespace with whitelisted builtins and imports."""
    safe_builtins = dict(SAFE_BUILTINS)
    safe_builtins['__import__'] = _safe_import

    namespace = {'__builtins__': safe_builtins}

    if function_library:
        for name, fn_code in function_library.items():
            try:
                # 1. Try to execute it (works for standard 'def func():' blocks)
                exec(fn_code, namespace)
                
                # 2. If it was a lambda, it didn't bind to the name automatically. 
                # Catch it and evaluate it so it gets bound to the namespace.
                if name not in namespace:
                    namespace[name] = eval(fn_code, namespace)
            except Exception as e:
                # 3. Stop failing silently so you can debug!
                print(f"Warning: Failed to load function '{name}' into sandbox: {e}")

    stdout_capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(compile(code, '<assistant>', 'exec'), namespace)
        output = stdout_capture.getvalue()
        return output if output else "(executed — no output)"
    except ImportError as e:
        return f"import blocked: {e}"
    except Exception as e:
        return f"error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# MEMORY (Supabase-backed)
# ─────────────────────────────────────────────────────────────────────────────

import os
from supabase import create_client, Client

def _get_supabase() -> Client:
    #url = os.environ.get("SUPABASE_URL") or os.environ.get("OUROBOROS_SUPABASE_URL")
    #key = os.environ.get("SUPABASE_KEY") or os.environ.get("OUROBOROS_SUPABASE_KEY")
    url = 'https://rbgomvgchwuynbxxiugt.supabase.co'
    key = 'sb_secret_TODQcDNzcxU9GtFnEJJTyg_rnptjwtz'
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set")
    return create_client(url, key)

class AssistantMemory:
    """Supabase-backed memory, functions, and conversation."""

    def __init__(self):
        self._db = _get_supabase()

    # ── Key-value memory ──

    def remember(self, key: str, value: str):
        self._db.table("assistant_memory").upsert({
            "key": key.strip(),
            "value": value.strip(),
            "updated_at": "now()"
        }).execute()

    def forget(self, key: str):
        self._db.table("assistant_memory").delete().eq("key", key.strip()).execute()

    def all_memories(self) -> dict:
        res = self._db.table("assistant_memory").select("key,value").order("key").execute()
        return {r["key"]: r["value"] for r in res.data}

    def memory_context(self) -> str:
        memories = self.all_memories()
        parts = []
        summary_ctx = self.summary_context()
        if summary_ctx:
            parts.append(summary_ctx)
        if memories:
            lines = [f"{k}: {v}" for k, v in memories.items()]
            parts.append("ASSISTANT'S MEMORY:\n" + "\n".join(lines))
        return "\n\n".join(parts)

    # ── Function library ──

    def write_function(self, name: str, code: str):
        self._db.table("assistant_functions").upsert({
            "name": name.strip(),
            "code": code.strip(),
            "updated_at": "now()"
        }).execute()

    def get_functions(self) -> dict:
        res = self._db.table("assistant_functions").select("name,code").execute()
        return {r["name"]: r["code"] for r in res.data}

    # ── Conversation ──

    def append_message(self, role: str, content: str):
        self._db.table("assistant_conversation").insert({
            "role": role,
            "content": content
        }).execute()

    def clear_conversation(self):
        self._db.table("assistant_conversation").delete().neq("id", 0).execute()

    # ── Cascade Search ──

    def update_memory_embedding(self, key: str, embedding: list):
        self._db.table("assistant_memory").update({"embedding": embedding
            
        }).eq("key", key).execute()

    def update_conversation_embedding(self, msg_id: int, embedding: list):
        self._db.table("assistant_conversation").update({
            "embedding": embedding
        }).eq("id", msg_id).execute()

    def cascade_search(self, query_embedding: list, threshold: float = 0.3, limit: int = 10) -> list:
        result = self._db.rpc('match_cascade', {
            'query_embedding': query_embedding,
            'match_threshold': threshold,
            'match_count': limit
        }).execute()
        
        return result.data if result.data else []

    def get_last_message_id(self) -> int:
        res = (self._db.table("assistant_conversation")
               .select("id")
               .order("id", desc=True)
               .limit(1)
               .execute())
        return res.data[0]["id"] if res.data else 0

    # ── Summaries (archive) ──

    MAX_CONTEXT_MESSAGES = 20   # messages held in the live context window
    SUMMARY_EVERY = 20          # summarise a batch after this many overflow messages
    MAX_SUMMARIES = 4

    def load_conversation(self, limit: int = None) -> list:
        """Load the most recent `limit` messages. Defaults to MAX_CONTEXT_MESSAGES."""
        n = limit if limit is not None else self.MAX_CONTEXT_MESSAGES
        res = (self._db.table("assistant_conversation")
               .select("role,content")
               .order("created_at", desc=True)
               .limit(n)
               .execute())
        rows = res.data or []
        rows.reverse()
        return [{"role": r["role"], "content": r["content"]} for r in rows]

    def count_conversation(self) -> int:
        """Total number of messages stored."""
        res = (self._db.table("assistant_conversation")
               .select("id", count="exact")
               .execute())
        return res.count or 0

    def get_messages_before_window(self) -> list:
        """Return all messages outside the active context window, oldest first."""
        total = self.count_conversation()
        overflow = total - self.MAX_CONTEXT_MESSAGES
        if overflow <= 0:
            return []
        res = (self._db.table("assistant_conversation")
               .select("id,role,content,created_at")
               .order("created_at", desc=False)
               .limit(overflow)
               .execute())
        return res.data or []

    def pending_summary_batch(self) -> list:
        """Return the oldest SUMMARY_EVERY unsummarised overflow messages, or []
        if there aren't enough to fill a full batch yet.

        A message is considered already summarised when its id appears in any
        covers_ids array stored in assistant_summaries.
        """
        overflow = self.get_messages_before_window()
        if not overflow:
            return []

        # Collect all message ids already covered by existing summaries
        covered = set()
        for s in self.get_summaries():
            for mid in (s.get("covers_ids") or []):
                covered.add(mid)

        unsummarised = [m for m in overflow if m["id"] not in covered]

        # Only trigger when we have a full batch ready
        if len(unsummarised) >= self.SUMMARY_EVERY:
            return unsummarised[: self.SUMMARY_EVERY]
        return []

    def update_summary_embedding(self, summary_id: int, embedding: list):
        """Store a vector embedding on a summary row for cascade search."""
        self._db.table("assistant_summaries").update({
            "embedding": embedding
        }).eq("id", summary_id).execute()

    def write_summary(self, summary_text: str, covers_message_ids: list) -> int:
        """Persist a summary. Rolls off the oldest if we're over MAX_SUMMARIES.
        Returns the id of the newly inserted summary row."""
        res = self._db.table("assistant_summaries").insert({
            "summary": summary_text,
            "covers_ids": covers_message_ids,
        }).execute()
        new_id = res.data[0]["id"] if res.data else None
        # Trim to MAX_SUMMARIES — delete oldest beyond the cap
        all_res = (self._db.table("assistant_summaries")
               .select("id")
               .order("created_at", desc=True)
               .execute())
        all_ids = [r["id"] for r in (all_res.data or [])]
        to_delete = all_ids[self.MAX_SUMMARIES:]
        for old_id in to_delete:
            self._db.table("assistant_summaries").delete().eq("id", old_id).execute()
        return new_id

    def get_summaries(self) -> list:
        """Return all summaries, oldest first."""
        res = (self._db.table("assistant_summaries")
               .select("id,summary,covers_ids,created_at")
               .order("created_at", desc=False)
               .execute())
        return res.data or []

    def delete_archived_messages(self, message_ids: list):
        """Remove messages that have been summarised and are safe to delete."""
        for mid in message_ids:
            self._db.table("assistant_conversation").delete().eq("id", mid).execute()

    def summary_context(self) -> str:
        """Build a human-readable block of summaries for the system prompt."""
        summaries = self.get_summaries()
        if not summaries:
            return ""
        lines = []
        for i, s in enumerate(summaries, 1):
            lines.append(f"[Archive {i}] {s['summary']}")
        return "CONVERSATION ARCHIVE (oldest → newest):\n" + "\n\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# ACTION ROUTER
# ─────────────────────────────────────────────────────────────────────────────

def route_tags(tags: dict, memory: AssistantMemory, search_fn=None) -> list:
    """
    Process parsed tags. Returns a list of events for the frontend.
    """
    events = []

    for thought in tags.get('think', []):
        events.append({'type': 'think', 'content': thought})

    for key, value in tags.get('remember', []):
        memory.remember(key, value)
        events.append({'type': 'remember', 'key': key, 'value': value})

    for key in tags.get('forget', []):
        memory.forget(key)
        events.append({'type': 'forget', 'key': key})

    for text in tags.get('say', []):
        events.append({'type': 'say', 'content': text})

    for action in tags.get('emote', []):
        events.append({'type': 'emote', 'content': action})

    for observation in tags.get('reflect', []):
        events.append({'type': 'reflect', 'content': observation})

    for query in tags.get('cascade', []):
        events.append({'type': 'cascade', 'query': query})

    for code in tags.get('execute', []):
        result = execute_python(code, memory.get_functions())
        events.append({'type': 'execute', 'code': code, 'result': result})

    for match in tags.get('write_function', []):
        name, code = match
        memory.write_function(name, code)
        events.append({'type': 'write_function', 'name': name, 'code': code})

    for invocation in tags.get('invoke', []):
        try:
            # Wrap the invocation to capture return value
            code = f"_result = {invocation}\nif _result is not None:\n    print(_result)"
            result = execute_python(code, memory.get_functions())
            events.append({'type': 'invoke', 'call': invocation, 'result': result})
        except Exception as e:
            events.append({'type': 'invoke', 'call': invocation, 'result': f'error: {e}'})
            
    return events