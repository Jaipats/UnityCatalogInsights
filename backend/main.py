"""
UnityCatalogInsights - FastAPI Backend
Answers natural language questions about Unity Catalog metadata using
Vector Search retrieval + Foundation Model API generation.

Uses Databricks Apps "on behalf of" authentication so each user's
UC permissions are respected — they only see metadata they have access to.
"""

import os
import json
import logging
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="UnityCatalogInsights", version="1.0.0")

# --- Configuration ---
_raw_host = os.environ.get("DATABRICKS_HOST", "")
DATABRICKS_HOST = _raw_host if _raw_host.startswith("https://") else f"https://{_raw_host}"
DATABRICKS_HOST = DATABRICKS_HOST.rstrip("/")
VS_ENDPOINT = os.environ.get("VS_ENDPOINT", "mas-587f2f3d-endpoint")
VS_INDEX = os.environ.get(
    "VS_INDEX",
    "experian_agent_demo_catalog.experian_agent_demo.uc_metadata_vs_index",
)
LLM_ENDPOINT = os.environ.get("LLM_ENDPOINT", "databricks-claude-sonnet-4")
CATALOG = os.environ.get("UC_CATALOG", "experian_agent_demo_catalog")
SCHEMA = os.environ.get("UC_SCHEMA", "experian_agent_demo")


# --- Authentication ---
# `valueFrom: me` injects a token scoped to the logged-in user.
# Use DATABRICKS_TOKEN env var for all Databricks API calls.
_DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")


def _api_headers(request: Request = None) -> dict:
    """Build headers for Databricks API calls using the app token."""
    return {
        "Authorization": f"Bearer {_DATABRICKS_TOKEN}",
        "Content-Type": "application/json",
    }


# --- Pydantic Models ---
class ChatRequest(BaseModel):
    message: str
    num_results: int = 5


class Source(BaseModel):
    table_name: str
    nl_description: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    user: Optional[str] = None


class TableDetail(BaseModel):
    table_name: str
    content: str
    nl_description: str


# --- User Identity ---
def get_user_info(request: Request) -> dict:
    """Get the current user's identity from Databricks Apps headers."""
    # Databricks Apps proxy sets these headers
    user_email = request.headers.get("X-Forwarded-Email", "")
    user_name = request.headers.get("X-Forwarded-User", "")
    return {
        "email": user_email,
        "name": user_name or user_email,
    }


# --- User-Visible Tables (ACL enforcement via INFORMATION_SCHEMA) ---
# Cache of user -> (timestamp, set of visible table names)
_user_acl_cache: dict[str, tuple[float, set[str]]] = {}
_ACL_CACHE_TTL = 300  # 5 minutes

WAREHOUSE_ID = os.environ.get("WAREHOUSE_ID", "")


def get_user_visible_tables(request: Request) -> set[str]:
    """
    Query INFORMATION_SCHEMA using the user's on-behalf-of token to get
    the set of tables they can actually see. INFORMATION_SCHEMA automatically
    filters by the querying user's UC permissions — this is the authoritative
    source of what a user has access to.

    Results are cached per-user for 5 minutes to avoid hammering the SQL API.
    """
    import time

    user = get_user_info(request)
    user_key = user.get("email", "unknown")
    now = time.time()

    # Check cache
    if user_key in _user_acl_cache:
        cached_time, cached_tables = _user_acl_cache[user_key]
        if now - cached_time < _ACL_CACHE_TTL:
            logger.debug(f"ACL cache hit for {user_key}: {len(cached_tables)} tables")
            return cached_tables

    # Query INFORMATION_SCHEMA as the user (on-behalf-of)
    sql = f"""
        SELECT CONCAT(table_catalog, '.', table_schema, '.', table_name) AS full_table_name
        FROM {CATALOG}.information_schema.tables
        WHERE table_schema != 'information_schema'
          AND table_name NOT LIKE 'mlflow_%'
          AND table_name NOT LIKE 'uc_metadata_%'
    """

    url = f"{DATABRICKS_HOST}/api/2.0/sql/statements"
    payload = {
        "statement": sql,
        "warehouse_id": WAREHOUSE_ID,
        "wait_timeout": "30s",
        "on_wait_timeout": "CANCEL",
    }

    resp = requests.post(url, headers=_api_headers(request), json=payload)

    if resp.status_code != 200:
        logger.warning(f"INFORMATION_SCHEMA query failed for {user_key}: {resp.status_code}")
        # On failure, fall back to allowing all results (fail-open for usability)
        # In production, you may want fail-closed instead
        return None

    data = resp.json()
    status = data.get("status", {}).get("state", "")

    if status != "SUCCEEDED":
        logger.warning(f"SQL statement did not succeed for {user_key}: {status}")
        return None

    # Extract table names from result
    visible_tables = set()
    result_data = data.get("result", {})
    for chunk in result_data.get("data_array", []):
        if chunk and chunk[0]:
            visible_tables.add(chunk[0])

    # Cache the result
    _user_acl_cache[user_key] = (now, visible_tables)
    logger.info(f"ACL check for {user_key}: {len(visible_tables)} visible tables")

    return visible_tables


# --- Vector Search + Post-Retrieval ACL Filtering ---
def search_uc_metadata(request: Request, query: str, num_results: int = 5) -> list[dict]:
    """
    Search the UC metadata Vector Search index, then filter results to only
    include tables the current user has access to (via INFORMATION_SCHEMA).

    Vector Search itself does NOT enforce row-level UC permissions —
    INFORMATION_SCHEMA is the authoritative filter.
    """
    # Step 1: Get user's visible tables from INFORMATION_SCHEMA
    visible_tables = get_user_visible_tables(request)

    # Step 2: Query Vector Search (fetch extra results to compensate for filtering)
    fetch_count = num_results * 3 if visible_tables is not None else num_results
    url = f"{DATABRICKS_HOST}/api/2.0/vector-search/indexes/{VS_INDEX}/query"
    payload = {
        "query_text": query,
        "columns": [
            "doc_id", "table_name", "nl_description", "content",
            "column_names", "tag_summary", "acl_summary",
        ],
        "num_results": fetch_count,
    }

    resp = requests.post(url, headers=_api_headers(request), json=payload)
    if resp.status_code == 403:
        logger.warning(f"User lacks permission to query Vector Search index")
        raise HTTPException(
            status_code=403,
            detail="You don't have permission to query this catalog index. Contact your admin.",
        )
    if resp.status_code != 200:
        logger.error(f"Vector Search error: {resp.status_code} {resp.text}")
        raise HTTPException(status_code=502, detail=f"Vector Search query failed: {resp.text}")

    data = resp.json()
    columns = data.get("manifest", {}).get("columns", [])
    col_names = [c["name"] for c in columns]

    results = []
    for row in data.get("result", {}).get("data_array", []):
        row_dict = dict(zip(col_names, row))
        results.append(row_dict)

    # Step 3: Filter by user's UC permissions
    if visible_tables is not None:
        filtered = [r for r in results if r.get("table_name") in visible_tables]
        removed = len(results) - len(filtered)
        if removed > 0:
            logger.info(f"ACL filter removed {removed} results the user cannot access")
        results = filtered[:num_results]
    else:
        # If ACL check failed, return unfiltered (fail-open)
        results = results[:num_results]

    return results


# --- LLM Generation (uses service principal for model serving) ---
SYSTEM_PROMPT = """You are a Unity Catalog Discovery Agent. You help users understand their data catalog by answering questions about tables, columns, schemas, data meaning, relationships, and usage patterns.

You have access to metadata from Unity Catalog including table structures, column definitions, AI-generated descriptions, tags, and access privileges.

When answering:
- Be specific about table names, column names, and data types
- Explain what the data means in business terms
- Suggest how tables relate to each other (e.g., via customer_id joins)
- If relevant, mention who has access (ACLs) and any tags
- If a user asks about a concept, map it to the actual tables/columns that contain that information
- Format your response clearly with markdown

Always ground your answers in the retrieved metadata. If the metadata doesn't contain relevant information, say so clearly."""


def generate_answer(request: Request, question: str, context_docs: list[dict], user_name: str) -> str:
    """Generate an answer using the Foundation Model API with retrieved context."""
    context_parts = []
    for i, doc in enumerate(context_docs, 1):
        context_parts.append(
            f"### Source {i}: {doc.get('table_name', 'Unknown')}\n"
            f"{doc.get('content', doc.get('nl_description', 'No description available'))}"
        )
    context = "\n\n---\n\n".join(context_parts)

    url = f"{DATABRICKS_HOST}/serving-endpoints/{LLM_ENDPOINT}/invocations"
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"User: {user_name}\n\n"
                    f"Here is the relevant Unity Catalog metadata:\n\n{context}\n\n---\n\n"
                    f"User question: {question}"
                ),
            },
        ],
        "max_tokens": 1500,
        "temperature": 0.1,
    }

    # Use user's token for model serving so usage is attributed to them
    resp = requests.post(url, headers=_api_headers(request), json=payload)
    if resp.status_code != 200:
        logger.error(f"LLM error: {resp.status_code} {resp.text}")
        raise HTTPException(status_code=502, detail=f"LLM generation failed: {resp.text}")

    return resp.json()["choices"][0]["message"]["content"]


# --- API Endpoints ---
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "host": DATABRICKS_HOST,
        "vs_index": VS_INDEX,
        "llm_endpoint": LLM_ENDPOINT,
        "token_present": bool(_DATABRICKS_TOKEN),
    }


@app.get("/api/me")
def get_me(request: Request):
    """Return the current authenticated user's identity."""
    user = get_user_info(request)
    return {"user": user}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    """Main chat endpoint: retrieves relevant UC metadata and generates an answer."""
    user = get_user_info(request)
    logger.info(f"Chat request from {user.get('email', 'unknown')}: {req.message}")

    # Step 1: Retrieve relevant metadata using user's token (ACL-filtered)
    search_results = search_uc_metadata(request, req.message, num_results=req.num_results)

    if not search_results:
        return ChatResponse(
            answer="I couldn't find any relevant tables in the catalog for your question. "
            "This could mean the tables don't exist, or you may not have access. "
            "Try rephrasing or asking about specific table names, columns, or data domains.",
            sources=[],
            user=user.get("email"),
        )

    # Step 2: Generate answer with LLM (on behalf of user)
    answer = generate_answer(request, req.message, search_results, user.get("name", ""))

    # Step 3: Build sources list
    sources = []
    for doc in search_results:
        score_val = doc.get("score", 0.0)
        sources.append(
            Source(
                table_name=doc.get("table_name", "Unknown"),
                nl_description=doc.get("nl_description", "")[:300],
                score=float(score_val) if score_val else 0.0,
            )
        )

    return ChatResponse(answer=answer, sources=sources, user=user.get("email"))


@app.get("/api/tables")
def list_tables(request: Request):
    """List all indexed tables visible to the current user."""
    results = search_uc_metadata(request, "list all tables", num_results=50)
    tables = []
    seen = set()
    for doc in results:
        name = doc.get("table_name", "")
        if name and name not in seen:
            seen.add(name)
            tables.append(
                {
                    "table_name": name,
                    "nl_description": doc.get("nl_description", "")[:200],
                    "column_names": doc.get("column_names", ""),
                    "tag_summary": doc.get("tag_summary", ""),
                }
            )
    return {"tables": tables}


@app.get("/api/table/{table_name:path}", response_model=TableDetail)
def get_table_detail(table_name: str, request: Request):
    """Get detailed metadata for a specific table (ACL-filtered)."""
    results = search_uc_metadata(request, f"table {table_name}", num_results=1)
    if not results:
        raise HTTPException(
            status_code=404,
            detail=f"Table '{table_name}' not found or you don't have access.",
        )
    doc = results[0]
    return TableDetail(
        table_name=doc.get("table_name", table_name),
        content=doc.get("content", ""),
        nl_description=doc.get("nl_description", ""),
    )


# --- Serve Frontend ---
static_dir = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")


@app.get("/")
def serve_index():
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.get("/{path:path}")
def serve_static(path: str):
    file_path = os.path.join(static_dir, path)
    if os.path.isfile(file_path):
        return FileResponse(file_path)
    return FileResponse(os.path.join(static_dir, "index.html"))
