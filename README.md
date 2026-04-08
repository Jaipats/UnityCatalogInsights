# UnityCatalogInsights

A Databricks App that lets users explore and understand their Unity Catalog metadata through natural language conversation. Ask questions like *"What tables contain customer credit information?"* or *"How can I join customer data with credit scores?"* and get grounded, context-aware answers with source references.

Descriptions are sourced directly from **user-authored UC comments** at the catalog, schema, table, and column levels — no LLM is used during indexing.

![Architecture](frontend/uc_discovery_agent_arch.png)

## Features

- **Natural Language Data Discovery** — Ask questions about your data catalog in plain English. The agent retrieves relevant table metadata via semantic search and generates detailed answers using an LLM.
- **User-Authored Descriptions** — Indexes existing comments from Unity Catalog at every level (catalog, schema, table, column) plus tags. No LLM-generated descriptions — what users documented is what gets indexed.
- **Post-Retrieval ACL Filtering** — Vector Search results are filtered against `INFORMATION_SCHEMA` using the logged-in user's on-behalf-of token, ensuring users only see metadata for tables they have access to.
- **Dual-Token Auth** — SP token for Vector Search and LLM endpoints; user's OBO token for SQL DW ACL queries. Each API call uses the appropriate identity.
- **Source Attribution** — Every answer includes source cards showing which tables were used, with relevance scores.
- **Sidebar Table Browser** — Browse all indexed tables in the sidebar and click to learn more about any table.

## Architecture

### Query Flow

```
User Question
    |
    v
[1] Vector Search (semantic retrieval — SP token)
    |
    v
[2] SQL Statement API → INFORMATION_SCHEMA (ACL check — user OBO token)
    |
    v
[3] Filter VS results to only tables the user can see
    |
    v
[4] Foundation Model API / Claude Sonnet 4 (generate answer — SP token)
    |
    v
Answer + Sources → User
```

### Data Pipeline (Offline Indexing)

```
Unity Catalog INFORMATION_SCHEMA
    |
    +-- Catalog comments (DESCRIBE CATALOG)
    +-- Schema comments (information_schema.schemata)
    +-- Table comments (information_schema.tables)
    +-- Column comments (information_schema.columns)
    +-- Tags (table_tags, column_tags)
    +-- ACLs (table_privileges)
    |
    v
Extraction Notebook (serverless compute) — no LLM calls
    |
    v
Delta Table (uc_metadata_documents, Change Data Feed enabled)
    |
    v
Vector Search Index (Delta Sync, BGE-Large embeddings)
```

### Authentication Model

| API Call | Token Used | Why |
|----------|-----------|-----|
| Vector Search query | **SP token** (PAT) | System-level index; ACL filtering is done post-retrieval |
| SQL Statement API (INFORMATION_SCHEMA) | **User OBO token** | Must run as the logged-in user so results reflect their UC grants |
| Foundation Model API (LLM) | **SP token** (PAT) | System-level serving endpoint |

## Project Structure

```
UnityCatalogInsights/
├── app.yaml                          # Databricks App configuration
├── requirements.txt                  # Python dependencies
├── backend/
│   ├── main.py                       # FastAPI backend (chat, search, ACL filtering)
│   └── requirements.txt              # Backend-specific dependencies
├── frontend/
│   ├── dist/
│   │   └── index.html                # Production frontend (self-contained, CDN-loaded)
│   └── static/
│       └── index.html                # Alternate static frontend
└── notebooks/
    └── 01_extract_and_index_uc_metadata.py   # Metadata extraction & VS indexing notebook
```

## Setup & Deployment

### Prerequisites

- A Databricks workspace with Unity Catalog enabled
- Tables with comments/descriptions you want to index
- A SQL Warehouse (for INFORMATION_SCHEMA ACL queries at runtime)
- Access to Foundation Model API endpoints (Claude Sonnet 4 for answers, BGE-Large for embeddings)
- An existing Vector Search endpoint in ONLINE state

### Step 1: Run the Metadata Extraction Notebook

1. Import `notebooks/01_extract_and_index_uc_metadata.py` into your workspace
2. Update the configuration variables at the top:
   ```python
   CATALOG = "your_catalog"
   SCHEMA = "your_schema"
   VS_ENDPOINT = "your-vs-endpoint"  # Must be an existing ONLINE endpoint
   ```
3. Attach to a cluster or run as a serverless job
4. Run All — the notebook will:
   - Extract comments at catalog, schema, table, and column levels from `INFORMATION_SCHEMA`
   - Collect tags and ACL/privilege information
   - Build searchable documents from existing UC metadata (no LLM calls)
   - Save to a Delta table with Change Data Feed enabled
   - Create or sync a Vector Search index with BGE-Large embeddings

### Step 2: Create and Deploy the App

```bash
# Create the app
databricks apps create unity-catalog-insights --profile=<your-profile>

# Sync source code to workspace
databricks sync . /Users/<you>/apps/unity-catalog-insights --profile=<your-profile> --full

# Deploy
databricks apps deploy unity-catalog-insights \
  --source-code-path /Workspace/Users/<you>/apps/unity-catalog-insights \
  --profile=<your-profile>
```

### Step 3: Configure Authentication

Update `app.yaml` with your environment:

```yaml
env:
  - name: DATABRICKS_HOST
    valueFrom: workspace-url
  - name: DATABRICKS_TOKEN
    value: "<your-pat-token>"        # PAT for VS + LLM access (SP token)
  - name: VS_ENDPOINT
    value: "<your-vs-endpoint>"
  - name: VS_INDEX
    value: "<catalog.schema.index_name>"
  - name: LLM_ENDPOINT
    value: "databricks-claude-sonnet-4"
  - name: WAREHOUSE_ID
    value: "<your-warehouse-id>"     # SQL warehouse for OBO ACL queries
```

The **user's OBO token** is automatically provided by the Databricks Apps proxy — no configuration needed for that.

### Step 4: Grant Permissions

The PAT owner (or service principal) needs:
- `USE_CATALOG` on the target catalog
- `USE_SCHEMA` and `SELECT` on the target schema
- Access to query the Vector Search endpoint
- Access to query the Foundation Model API serving endpoints
- Membership in the workspace `users` group (for Foundation Model API access)

## How ACL Filtering Works

Vector Search indexes are static snapshots — they don't enforce per-user Unity Catalog permissions at query time. To prevent metadata leakage:

1. **At query time**, the backend queries `INFORMATION_SCHEMA.TABLES` via the SQL Statement API using the **user's on-behalf-of token** (not the SP token)
2. INFORMATION_SCHEMA automatically filters results to only tables that specific user has UC grants for
3. **Post-retrieval filter** removes any Vector Search results for tables the user can't see
4. **Only permitted results** are sent to the LLM for answer generation
5. Results are **cached per-user for 5 minutes** to avoid excessive SQL calls

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla JS + React (CDN) + Marked.js |
| Search | Mosaic AI Vector Search (Delta Sync, BGE-Large embeddings) |
| LLM (answers only) | Foundation Model API (Claude Sonnet 4) |
| ACL Enforcement | INFORMATION_SCHEMA + SQL Statement API (user OBO token) |
| Deployment | Databricks Apps |
| Data Storage | Delta Lake (with Change Data Feed) |
| Metadata Source | Unity Catalog comments (catalog, schema, table, column) + tags |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check with config details |
| `/api/me` | GET | Current user identity (from OBO headers) |
| `/api/chat` | POST | Main chat — takes `{message, num_results}`, returns `{answer, sources}` |
| `/api/tables` | GET | List all indexed tables visible to current user |
| `/api/table/{name}` | GET | Get detailed metadata for a specific table |

## License

Internal use only.
