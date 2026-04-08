# UnityCatalogInsights

An AI-powered Databricks App that lets users explore and understand their Unity Catalog metadata through natural language conversation. Ask questions like *"What tables contain customer credit information?"* or *"How can I join customer data with credit scores?"* and get grounded, context-aware answers with source references.

![Architecture](frontend/uc_discovery_agent_arch.png)

## Features

- **Natural Language Data Discovery** — Ask questions about your data catalog in plain English. The agent retrieves relevant table metadata via semantic search and generates detailed answers using an LLM.
- **AI-Enriched Metadata** — Each table's metadata is automatically enriched with LLM-generated natural language descriptions that explain what the data means, its business context, and how it relates to other tables.
- **Post-Retrieval ACL Filtering** — Vector Search results are filtered against `INFORMATION_SCHEMA` using the caller's permissions, ensuring users only see metadata for tables they have access to in Unity Catalog.
- **Source Attribution** — Every answer includes source cards showing which tables were used, with relevance scores.
- **Sidebar Table Browser** — Browse all indexed tables in the sidebar and click to learn more about any table.

## Architecture

### Query Flow

```
User Question
    |
    v
[1] Vector Search (semantic retrieval from uc_metadata_vs_index)
    |
    v
[2] INFORMATION_SCHEMA query (ACL filter — user's visible tables)
    |
    v
[3] Filter VS results to only permitted tables
    |
    v
[4] Foundation Model API (Claude Sonnet 4 — generate answer)
    |
    v
Answer + Sources → User
```

### Data Pipeline (Offline Indexing)

```
INFORMATION_SCHEMA (tables, columns, tags, ACLs)
    |
    v
Extraction Notebook (serverless compute)
    |
    v
LLM Enrichment (generate NL descriptions per table)
    |
    v
Delta Table (uc_metadata_documents, with Change Data Feed)
    |
    v
Vector Search Index (Delta Sync pipeline, BGE-Large embeddings)
```

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
- Tables with metadata you want to index
- A SQL Warehouse (for INFORMATION_SCHEMA queries)
- Access to Foundation Model API endpoints (Claude Sonnet 4, BGE-Large)

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
   - Extract table/column metadata, tags, and ACLs from `INFORMATION_SCHEMA`
   - Generate AI-powered natural language descriptions for each table
   - Save enriched documents to a Delta table
   - Create a Vector Search index with Delta Sync

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
    value: "<your-pat-token>"        # PAT with access to VS, serving endpoints, SQL
  - name: VS_ENDPOINT
    value: "<your-vs-endpoint>"
  - name: VS_INDEX
    value: "<catalog.schema.index_name>"
  - name: LLM_ENDPOINT
    value: "databricks-claude-sonnet-4"
  - name: WAREHOUSE_ID
    value: "<your-warehouse-id>"     # For INFORMATION_SCHEMA ACL queries
```

### Step 4: Grant Permissions

The token owner (or service principal) needs:
- `USE_CATALOG` on the target catalog
- `USE_SCHEMA` and `SELECT` on the target schema
- Access to query the Vector Search endpoint
- Access to query the Foundation Model API serving endpoints
- Membership in the workspace `users` group (for Foundation Model API access)

## How ACL Filtering Works

Vector Search indexes are static snapshots — they don't enforce per-user Unity Catalog permissions at query time. To prevent metadata leakage:

1. **At query time**, the backend queries `INFORMATION_SCHEMA.TABLES` using the SQL Statement API to get the set of tables the current user can see
2. **Post-retrieval filter** removes any Vector Search results for tables the user doesn't have access to
3. **Only permitted results** are sent to the LLM for answer generation
4. Results are **cached per-user for 5 minutes** to avoid excessive SQL calls

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla JS + React (CDN) + Marked.js |
| Search | Mosaic AI Vector Search (Delta Sync, BGE-Large embeddings) |
| LLM | Foundation Model API (Claude Sonnet 4) |
| ACL Enforcement | INFORMATION_SCHEMA + SQL Statement API |
| Deployment | Databricks Apps |
| Data Storage | Delta Lake (with Change Data Feed) |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check with config details |
| `/api/me` | GET | Current user identity |
| `/api/chat` | POST | Main chat — takes `{message, num_results}`, returns `{answer, sources}` |
| `/api/tables` | GET | List all indexed tables |
| `/api/table/{name}` | GET | Get detailed metadata for a specific table |

## License

Internal use only.
