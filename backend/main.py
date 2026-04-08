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
    # x-forwarded-user is "userId@orgId" — extract the user ID
    forwarded_user = request.headers.get("X-Forwarded-User", "")
    user_id = forwarded_user.split("@")[0] if "@" in forwarded_user else forwarded_user
    return {
        "email": user_email,
        "name": user_name or user_email,
        "user_id": user_id,
    }


# --- Per-Table ACL Check via SQL DW ---
WAREHOUSE_ID = os.environ.get("WAREHOUSE_ID", "")


def _run_sql(sql: str) -> dict:
    """Execute SQL via Statement API using SP token. Returns the response JSON."""
    url = f"{DATABRICKS_HOST}/api/2.0/sql/statements"
    payload = {
        "statement": sql,
        "warehouse_id": WAREHOUSE_ID,
        "wait_timeout": "30s",
        "on_wait_timeout": "CANCEL",
    }
    resp = requests.post(url, headers=_sp_headers(), json=payload)
    if resp.status_code != 200:
        logger.error(f"SQL API error: {resp.status_code} {resp.text}")
        return {}
    return resp.json()


def check_user_table_access(request: Request, table_names: list[str]) -> set[str]:
    """
    For each table returned by Vector Search, check if the logged-in user
    has access by running SHOW GRANTS on each table via SQL DW and looking
    for the user's email or ID in the results.

    Returns the subset of table_names the user can access.
    """
    if not table_names:
        return set()

    user = get_user_info(request)
    user_email = user.get("email", "")
    user_id = user.get("user_id", "")

    if not user_email:
        logger.error("No user email in request headers — denying access")
        return set()

    logger.info(f"Checking access for {user_email} (id={user_id}) on {len(table_names)} tables")

    accessible = set()
    for table_name in table_names:
        try:
            sql = f"SHOW GRANTS ON TABLE {table_name}"
            data = _run_sql(sql)

            if data.get("status", {}).get("state") != "SUCCEEDED":
                logger.warning(f"SHOW GRANTS failed for {table_name}: {data.get('status', {})}")
                continue

            # Check if user's email, ID, or a group they belong to has a grant
            for row in data.get("result", {}).get("data_array", []):
                # SHOW GRANTS returns: [principal, action_type, object_type, object_key]
                if not row or len(row) < 2:
                    continue
                grantee = str(row[0]).lower()
                if (
                    grantee == user_email.lower()
                    or grantee == user_id
                    or grantee in ("account users", "users")
                ):
                    accessible.add(table_name)
                    logger.debug(f"  {table_name}: ACCESS GRANTED (grantee={row[0]}, action={row[1]})")
                    break
            else:
                logger.debug(f"  {table_name}: ACCESS DENIED for {user_email}")

        except Exception as e:
            logger.error(f"Error checking grants on {table_name}: {e}")
            continue

    logger.info(f"Access check result: {len(accessible)}/{len(table_names)} tables accessible")
    return accessible


# --- Vector Search + Per-Table ACL Filtering ---
def search_uc_metadata(request: Request, query: str, num_results: int = 5) -> list[dict]:
    """
    1. Query Vector Search for semantically relevant tables
    2. For each table, check if the user has access via SHOW GRANTS
    3. Only return tables the user can access
    """
    # Step 1: Query Vector Search
    fetch_count = num_results * 3  # Fetch extra to compensate for filtering
    url = f"{DATABRICKS_HOST}/api/2.0/vector-search/indexes/{VS_INDEX}/query"
    payload = {
        "query_text": query,
        "columns": [
            "doc_id", "table_name", "nl_description", "content",
            "column_names", "tag_summary", "acl_summary",
        ],
        "num_results": fetch_count,
    }

    resp = requests.post(url, headers=_sp_headers(), json=payload)
    if resp.status_code == 403:
        raise HTTPException(status_code=403, detail="No permission to query the catalog index.")
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

    if not results:
        return []

    # Step 2: Check each table's access via SHOW GRANTS
    table_names = list({r.get("table_name") for r in results if r.get("table_name")})
    accessible_tables = check_user_table_access(request, table_names)

    if not accessible_tables:
        logger.warning("User has no access to any returned tables")
        return []

    # Step 3: Filter results to only accessible tables
    filtered = [r for r in results if r.get("table_name") in accessible_tables]
    removed = len(results) - len(filtered)
    if removed > 0:
        logger.info(f"ACL filter removed {removed}/{len(results)} results")

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
