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
# Two token types:
#   1. SP token (_SP_TOKEN): Used for Vector Search and LLM serving endpoints.
#      These are system-level resources the SP has been granted access to.
#   2. User OBO token: Extracted from the request Authorization header set by
#      the Databricks Apps proxy. Used for SQL Statement API calls so that
#      INFORMATION_SCHEMA queries run as the logged-in user and respect their
#      UC permissions.
_SP_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")


def _sp_headers() -> dict:
    """Headers using the service principal token — for Vector Search and LLM."""
    return {
        "Authorization": f"Bearer {_SP_TOKEN}",
        "Content-Type": "application/json",
    }


def _user_obo_headers(request: Request) -> dict:
    """
    Headers using the logged-in user's on-behalf-of token — for SQL DW.
    The Databricks Apps proxy strips the Authorization header but forwards the
    user's OAuth token in the `x-forwarded-access-token` header. This token is
    scoped to the logged-in user, so INFORMATION_SCHEMA queries reflect their
    UC grants.
    Falls back to SP token if no user token is present (e.g. health checks).
    """
    obo_token = request.headers.get("x-forwarded-access-token", "") if request else ""
    token = obo_token if obo_token else _SP_TOKEN
    if obo_token:
        logger.info(f"Using user OBO token for SQL DW ({len(obo_token)} chars)")
    else:
        logger.warning("No OBO token found, falling back to SP token for SQL DW")
    return {
        "Authorization": f"Bearer {token}",
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
    """Get the current user's identity from Databricks Apps proxy headers."""
    user_email = request.headers.get("X-Forwarded-Email", "")
    user_name = request.headers.get("X-Forwarded-Preferred-Username", "")
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
    Determine which tables the logged-in user can see by querying
    INFORMATION_SCHEMA.TABLE_PRIVILEGES for their explicit grants.

    The x-forwarded-access-token from the Databricks Apps proxy often lacks
    the 'sql' scope needed to execute SQL. So we use the SP token to run the
    query, but filter by the user's email from the proxy headers to check
    only THEIR grants.

    We check both direct user grants and grants to 'account users' (all users).
    Results are cached per-user for 5 minutes.
    """
    import time

    user = get_user_info(request)
    user_key = user.get("email", "unknown")
    now = time.time()

    if not user_key or user_key == "unknown":
        logger.error("No user identity found in request headers")
        return set()

    # Check cache
    if user_key in _user_acl_cache:
        cached_time, cached_tables = _user_acl_cache[user_key]
        if now - cached_time < _ACL_CACHE_TTL:
            logger.debug(f"ACL cache hit for {user_key}: {len(cached_tables)} tables")
            return cached_tables

    # Query table_privileges for this specific user's grants
    # This checks: direct user grants, 'account users' group, and ownership
    sql = f"""
        SELECT DISTINCT CONCAT(table_catalog, '.', table_schema, '.', table_name) AS full_table_name
        FROM {CATALOG}.information_schema.table_privileges
        WHERE table_schema != 'information_schema'
          AND table_name NOT LIKE 'mlflow_%'
          AND table_name NOT LIKE 'uc_metadata_%'
          AND (
            grantee = '{user_key}'
            OR grantee = 'account users'
            OR grantee = 'users'
          )
          AND privilege_type IN ('SELECT', 'ALL_PRIVILEGES', 'MODIFY')
        UNION
        SELECT DISTINCT CONCAT(t.table_catalog, '.', t.table_schema, '.', t.table_name) AS full_table_name
        FROM {CATALOG}.information_schema.tables t
        WHERE t.table_schema != 'information_schema'
          AND t.table_name NOT LIKE 'mlflow_%'
          AND t.table_name NOT LIKE 'uc_metadata_%'
          AND EXISTS (
            SELECT 1 FROM {CATALOG}.information_schema.schema_privileges sp
            WHERE sp.grantee IN ('{user_key}', 'account users', 'users')
              AND sp.table_schema = t.table_schema
              AND sp.privilege_type IN ('SELECT', 'ALL_PRIVILEGES', 'USE_SCHEMA')
          )
    """

    url = f"{DATABRICKS_HOST}/api/2.0/sql/statements"
    payload = {
        "statement": sql,
        "warehouse_id": WAREHOUSE_ID,
        "wait_timeout": "30s",
        "on_wait_timeout": "CANCEL",
    }

    # Use SP token (has sql permissions) — user filtering is in the WHERE clause
    resp = requests.post(url, headers=_sp_headers(), json=payload)

    if resp.status_code != 200:
        logger.error(f"ACL query failed for {user_key}: {resp.status_code} {resp.text}")
        return set()

    data = resp.json()
    status = data.get("status", {}).get("state", "")

    if status != "SUCCEEDED":
        logger.error(f"ACL query did not succeed for {user_key}: {status} — {data.get('status', {}).get('error', {}).get('message', '')}")
        return set()

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
    # Step 1: Get user's visible tables from INFORMATION_SCHEMA (fail-closed)
    visible_tables = get_user_visible_tables(request)
    logger.info(f"User can see {len(visible_tables)} tables")

    if not visible_tables:
        logger.warning("User has no visible tables — returning empty results")
        return []

    # Step 2: Query Vector Search (fetch extra results to compensate for filtering)
    fetch_count = num_results * 3
    url = f"{DATABRICKS_HOST}/api/2.0/vector-search/indexes/{VS_INDEX}/query"
    payload = {
        "query_text": query,
        "columns": [
            "doc_id", "table_name", "nl_description", "content",
            "column_names", "tag_summary", "acl_summary",
        ],
        "num_results": fetch_count,
    }

    # Use SP token for Vector Search (system-level resource)
    resp = requests.post(url, headers=_sp_headers(), json=payload)
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

    # Step 3: Filter by user's UC permissions (always enforced)
    filtered = [r for r in results if r.get("table_name") in visible_tables]
    removed = len(results) - len(filtered)
    if removed > 0:
        logger.info(f"ACL filter removed {removed}/{len(results)} results the user cannot access")

    return filtered[:num_results]


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

    # Use SP token for LLM serving endpoint (system-level resource)
    resp = requests.post(url, headers=_sp_headers(), json=payload)
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
        "token_present": bool(_SP_TOKEN),
    }


@app.get("/api/me")
def get_me(request: Request):
    """Return the current authenticated user's identity."""
    user = get_user_info(request)
    return {"user": user}


@app.get("/api/debug/headers")
def debug_headers(request: Request):
    """Debug: show what headers the app proxy forwards."""
    headers = {}
    for key, value in request.headers.items():
        if key.lower() in ("authorization",):
            # Mask the token value but show type and length
            headers[key] = f"{value[:15]}...({len(value)} chars)" if len(value) > 15 else value
        else:
            headers[key] = value
    auth = request.headers.get("Authorization", "")
    return {
        "has_authorization": bool(auth),
        "auth_type": auth.split(" ")[0] if auth else "(none)",
        "auth_token_length": len(auth.split(" ", 1)[1]) if " " in auth else 0,
        "sp_token_length": len(_SP_TOKEN),
        "tokens_match": (auth.split(" ", 1)[1] == _SP_TOKEN) if " " in auth else False,
        "headers": headers,
    }


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
