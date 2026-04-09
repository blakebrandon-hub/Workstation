# Workstation

A personal AI assistant platform combining a sandboxed Python execution environment with a Gemini-powered conversational assistant. Built with Flask and backed by Supabase.

---

## Overview

Workstation has two distinct but integrated layers:

**The Workstation** is a manual, API-driven workspace: a Python sandbox with persistent key-value memory and a reusable function library. You interact with it directly via REST endpoints — run code, store facts, define functions, invoke them later.

**The Assistant** is the AI layer: a Gemini-powered conversational agent that lives on top of the workstation. It can think, remember, write and run code, reflect on results, and search its own memory — all expressed through a structured tag protocol embedded in its responses.

---

## Architecture

```
index.html          — Browser UI
server.py           — Flask app: routes for both the workstation and the assistant
engine.py           — Workspace class: sandbox execution + Supabase memory
assistant.py        — Assistant: system prompt, tag parser, memory, action router
```

**External services:**
- [Google Gemini](https://ai.google.dev/) — language model (`gemini-3.1-flash-lite-preview`) and embeddings (`text-embedding-004`)
- [Supabase](https://supabase.com/) — persistent storage for memory, functions, conversation history, and summaries

---

## Features

### Python Sandbox
Code runs in a restricted namespace with a curated set of safe builtins and a whitelist of allowed packages (including `numpy`, `pandas`, `requests`, `bs4`, `scipy`, and standard library modules). No arbitrary imports or unsafe builtins.

### Workstation Memory
A simple key-value store backed by Supabase. Store, retrieve, and delete named facts. Memory is injected into the sandbox namespace as `_memory` at runtime, so code can read it directly.

### Function Library
Define named Python functions that persist in Supabase. They're automatically loaded into the sandbox on every execution, so you can build up a personal toolkit over time.

### Assistant — AI Layer
The assistant communicates via a tag-based protocol. Its raw output is parsed for structured tags, which are routed to real actions:

| Tag | Effect |
|---|---|
| `think: [thought]` | Internal reasoning (visible in events, not shown to user) |
| `remember: [key] = [value]` | Persists a memory entry |
| `forget: [key]` | Deletes a memory entry |
| `execute: [python code]` | Runs code in the sandbox; result is fed back |
| `write_function: [name] \| [code]` | Saves a reusable function |
| `invoke: [function_name(args)]` | Calls a saved function |
| `reflect: [observation]` | Internal observation logged as an event |
| `cascade: [query]` | Semantic memory search |

### Long-term Memory (Summarization + Embeddings)
The assistant's conversation is kept at a rolling window of 20 messages. When older messages overflow, they're batched and summarized automatically using Gemini (every 20 messages), and up to 4 summaries are kept. Summaries are embedded with `text-embedding-004` and stored for semantic (cascade) search.

---

## API Reference

### Workstation — Memory
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/memory` | List all memory entries |
| `POST` | `/api/memory` | Set a key-value pair (`{key, value}`) |
| `GET` | `/api/memory/<key>` | Get a single entry |
| `DELETE` | `/api/memory/<key>` | Delete an entry |

### Workstation — Functions
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/functions` | List all saved functions |
| `POST` | `/api/functions` | Save a function (`{name, code}`) |
| `DELETE` | `/api/functions/<name>` | Delete a function |

### Workstation — Execution
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/run` | Execute Python code (`{code}`) |
| `POST` | `/api/invoke` | Invoke a saved function call (`{call}`) |

### Workstation — State
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/state` | Full snapshot: memories, functions, recent log |
| `POST` | `/api/log/clear` | Clear the execution log |

### Assistant
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/chat` | Send a message (`multipart/form-data`: `message`, optional `file` or `image_b64`) |
| `GET` | `/api/assistant/state` | Assistant memories, functions, summaries, and history length |
| `GET` | `/api/context` | Full context: system prompt, conversation, memory, summaries |
| `GET` | `/api/conversation` | Raw conversation history from Supabase |
| `POST` | `/api/cascade` | Semantic search (`{embedding, limit, threshold}`) |
| `POST` | `/api/embed` | Update an embedding on a memory or conversation row |

### Shared
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check; returns current model name |

---

## Setup

**Requirements:** Python 3.11+

```bash
pip install flask flask-cors google-genai supabase
```

**Configure credentials** in `engine.py` and `assistant.py` (or move to environment variables — the commented-out env var block in `assistant.py` shows the intended pattern):

```python
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-key
GEMINI_API_KEY=your-gemini-key
```

**Run:**

```bash
python server.py
# → http://localhost:7000
```

---

## Supabase Schema

Run the following SQL in your Supabase project's SQL editor to create all required tables and the cascade search function.

### Enable pgvector

```sql
create extension if not exists vector;
```

### Workstation Tables

```sql
-- Key-value memory for the workstation
create table workstation_memory (
    key text primary key,
    value text not null,
    updated_at timestamptz default now()
);

-- Reusable function library for the workstation
create table workstation_functions (
    name text primary key,
    code text not null,
    updated_at timestamptz default now()
);

-- Execution log
create table workstation_conversation (
    id bigint generated always as identity primary key,
    role text not null,
    content text not null,
    created_at timestamptz default now()
);
```

### Assistant Tables

```sql
-- Key-value memory for the assistant (with embedding for cascade search)
create table assistant_memory (
    key text primary key,
    value text not null,
    embedding vector(768),
    updated_at timestamptz default now()
);

-- Reusable function library for the assistant
create table assistant_functions (
    name text primary key,
    code text not null,
    updated_at timestamptz default now()
);

-- Conversation history (with embedding for cascade search)
create table assistant_conversation (
    id bigint generated always as identity primary key,
    role text not null,
    content text not null,
    embedding vector(768),
    created_at timestamptz default now()
);

-- Compressed conversation archives (with embedding for cascade search)
create table assistant_summaries (
    id bigint generated always as identity primary key,
    summary text not null,
    covers_ids bigint[] not null default '{}',
    embedding vector(768),
    created_at timestamptz default now()
);
```

### Cascade Search Function

The cascade search performs a semantic similarity search across memory, conversation, and summaries simultaneously using pgvector.

```sql
create or replace function match_cascade(
    query_embedding vector(768),
    match_threshold float,
    match_count int
)
returns table (
    source text,
    content text,
    similarity float
)
language sql stable
as $$
    select 'memory' as source, key || ': ' || value as content,
        1 - (embedding <=> query_embedding) as similarity
    from assistant_memory
    where embedding is not null
        and 1 - (embedding <=> query_embedding) > match_threshold

    union all

    select 'conversation' as source, content,
        1 - (embedding <=> query_embedding) as similarity
    from assistant_conversation
    where embedding is not null
        and 1 - (embedding <=> query_embedding) > match_threshold

    union all

    select 'summary' as source, summary as content,
        1 - (embedding <=> query_embedding) as similarity
    from assistant_summaries
    where embedding is not null
        and 1 - (embedding <=> query_embedding) > match_threshold

    order by similarity desc
    limit match_count;
$$;
```

---

## Security Notes

- **API keys are currently hardcoded** in `engine.py` and `assistant.py`. Move them to environment variables before any deployment.
- The Python sandbox uses a whitelist of safe builtins and allowed packages, but `open` and `subprocess` are included — tighten these if deploying in a multi-user context.
- CORS is enabled globally via `flask-cors`. Restrict origins for production use.
