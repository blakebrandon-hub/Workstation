"""
Workspace Engine — memory, functions, sandbox. No AI.
"""

import io
import re
import sys
import contextlib
from supabase import create_client, Client

# ─────────────────────────────────────────────────────────────────────────────
# SUPABASE
# ─────────────────────────────────────────────────────────────────────────────

def _get_db() -> Client:
    SUPABASE_URL = os.environ.get("SUPABASE_URL")
    SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
    return create_client(url, key)


# ─────────────────────────────────────────────────────────────────────────────
# SANDBOX
# ─────────────────────────────────────────────────────────────────────────────

ALLOWED_PACKAGES = {
    'math', 'random', 'statistics', 'itertools', 'functools', 'time',
    'collections', 'datetime', 'json', 're', 'string', 'textwrap',
    'decimal', 'fractions', 'heapq', 'bisect', 'copy', 'sys', 'os',
    'numpy', 'pandas', 'scipy', 'matplotlib', 'requests', 'bs4',
    'difflib', 'unicodedata', 'hashlib', 'base64', 'urllib', 'subprocess',
    'ddgs', 'webbrowser', 'requests'
}

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
    'NotImplementedError': NotImplementedError, 'AttributeError': AttributeError,
}

def _safe_import(name, *args, **kwargs):
    base = name.split('.')[0]
    if base not in ALLOWED_PACKAGES:
        raise ImportError(f"'{name}' is not available in the sandbox")
    return __import__(name, *args, **kwargs)

def execute_python(code: str, function_library: dict = None, memory: dict = None) -> str:
    safe_builtins = dict(SAFE_BUILTINS)
    safe_builtins['__import__'] = _safe_import

    namespace = {'__builtins__': safe_builtins}

    if memory:
        namespace['_memory'] = dict(memory)

    if function_library:
        for name, fn_code in function_library.items():
            try:
                exec(fn_code, namespace)
                if name not in namespace:
                    namespace[name] = eval(fn_code, namespace)
            except Exception as e:
                pass

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_capture):
            with contextlib.redirect_stderr(stderr_capture):
                exec(compile(code, '<workspace>', 'exec'), namespace)
        output = stdout_capture.getvalue()
        err = stderr_capture.getvalue()
        if err:
            return f"{output}\n[stderr] {err}".strip()
        return output if output else "(executed — no output)"
    except ImportError as e:
        return f"ImportError: {e}"
    except SyntaxError as e:
        return f"SyntaxError: {e}"
    except Exception as e:
        return f"{type(e).__name__}: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# WORKSPACE MEMORY
# ─────────────────────────────────────────────────────────────────────────────

class Workspace:
    def __init__(self):
        self._db = _get_db()

    # ── Key-value memory ──

    def remember(self, key: str, value: str):
        self._db.table("workstation_memory").upsert({
            "key": key.strip(),
            "value": value.strip(),
            "updated_at": "now()"
        }).execute()

    def forget(self, key: str):
        self._db.table("workstation_memory").delete().eq("key", key.strip()).execute()

    def recall(self, key: str) -> str | None:
        res = self._db.table("workstation_memory").select("value").eq("key", key.strip()).maybe_single().execute()
        return res.data["value"] if res.data else None

    def all_memories(self) -> dict:
        res = self._db.table("workstation_memory").select("key,value").order("key").execute()
        return {r["key"]: r["value"] for r in (res.data or [])}

    # ── Function library ──

    def write_function(self, name: str, code: str):
        # Fixed typo: was "workstworkstation_functions"
        self._db.table("workstation_functions").upsert({
            "name": name.strip(),
            "code": code.strip(),
            "updated_at": "now()"
        }).execute()

    def delete_function(self, name: str):
        self._db.table("workstation_functions").delete().eq("name", name.strip()).execute()

    def get_functions(self) -> dict:
        res = self._db.table("workstation_functions").select("name,code").execute()
        return {r["name"]: r["code"] for r in (res.data or [])}

    # ── Log ──

    def log_entry(self, kind: str, content: str, result: str = None):
        entry = {"kind": kind, "content": content}
        if result is not None:
            entry["result"] = result
        self._db.table("workstation_conversation").insert({
            "role": kind,
            "content": content if result is None else f"{content}\n---\n{result}"
        }).execute()

    def load_log(self, limit: int = 100) -> list:
        res = (self._db.table("workstation_conversation")
               .select("id,role,content,created_at")
               .order("created_at", desc=True)
               .limit(limit)
               .execute())
        rows = res.data or []
        rows.reverse()
        return rows

    def clear_log(self):
        self._db.table("workstation_conversation").delete().neq("id", 0).execute()

    # ── Execute with logging ──

    def run(self, code: str) -> str:
        fns = self.get_functions()
        mem = self.all_memories()
        result = execute_python(code, fns, mem)
        self.log_entry("run", code, result)
        return result

    def invoke(self, fn_call: str) -> str:
        code = f"_r = {fn_call}\nif _r is not None: print(_r)"
        fns = self.get_functions()
        mem = self.all_memories()
        result = execute_python(code, fns, mem)
        self.log_entry("invoke", fn_call, result)
        return result