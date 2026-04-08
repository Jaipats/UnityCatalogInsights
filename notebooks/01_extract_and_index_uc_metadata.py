# Databricks notebook source
# MAGIC %md
# MAGIC # UC Metadata Extraction & Vector Search Indexing
# MAGIC
# MAGIC This notebook extracts Unity Catalog metadata (tables, columns, descriptions, tags, ACLs)
# MAGIC and indexes it into a Mosaic AI Vector Search index for natural language discovery.

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch databricks-sdk --upgrade -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import json
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, LongType
from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

CATALOG = "experian_agent_demo_catalog"
SCHEMA = "experian_agent_demo"
METADATA_TABLE = f"{CATALOG}.{SCHEMA}.uc_metadata_documents"
VS_ENDPOINT = "mas-587f2f3d-endpoint"
VS_INDEX = f"{CATALOG}.{SCHEMA}.uc_metadata_vs_index"

w = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Extract UC Metadata from INFORMATION_SCHEMA

# COMMAND ----------

# Get all tables with their comments
tables_df = spark.sql(f"""
    SELECT
        table_catalog,
        table_schema,
        table_name,
        CONCAT(table_catalog, '.', table_schema, '.', table_name) AS full_table_name,
        table_type,
        comment AS table_comment,
        created,
        last_altered
    FROM {CATALOG}.information_schema.tables
    WHERE table_schema != 'information_schema'
      AND table_name NOT LIKE 'mlflow_%'
""")

print(f"Found {tables_df.count()} tables")
tables_df.display()

# COMMAND ----------

# Get all columns with their comments
columns_df = spark.sql(f"""
    SELECT
        table_catalog,
        table_schema,
        table_name,
        CONCAT(table_catalog, '.', table_schema, '.', table_name) AS full_table_name,
        column_name,
        data_type,
        ordinal_position,
        is_nullable,
        comment AS column_comment
    FROM {CATALOG}.information_schema.columns
    WHERE table_schema != 'information_schema'
      AND table_name NOT LIKE 'mlflow_%'
    ORDER BY table_catalog, table_schema, table_name, ordinal_position
""")

print(f"Found {columns_df.count()} columns")
columns_df.display()

# COMMAND ----------

# Get tags if available
try:
    table_tags_df = spark.sql(f"""
        SELECT
            catalog_name,
            schema_name,
            table_name,
            CONCAT(catalog_name, '.', schema_name, '.', table_name) AS full_table_name,
            tag_name,
            tag_value
        FROM {CATALOG}.information_schema.table_tags
        WHERE schema_name != 'information_schema'
          AND table_name NOT LIKE 'mlflow_%'
    """)
    has_tags = table_tags_df.count() > 0
    print(f"Found {table_tags_df.count()} table tags")
except Exception as e:
    print(f"No table tags available: {e}")
    table_tags_df = None
    has_tags = False

try:
    column_tags_df = spark.sql(f"""
        SELECT
            catalog_name,
            schema_name,
            table_name,
            CONCAT(catalog_name, '.', schema_name, '.', table_name) AS full_table_name,
            column_name,
            tag_name,
            tag_value
        FROM {CATALOG}.information_schema.column_tags
        WHERE schema_name != 'information_schema'
          AND table_name NOT LIKE 'mlflow_%'
    """)
    has_col_tags = column_tags_df.count() > 0
    print(f"Found {column_tags_df.count()} column tags")
except Exception as e:
    print(f"No column tags available: {e}")
    column_tags_df = None
    has_col_tags = False

# COMMAND ----------

# Get ACLs / privileges
privileges_df = spark.sql(f"""
    SELECT
        grantor,
        grantee,
        table_catalog,
        table_schema,
        table_name,
        CONCAT(table_catalog, '.', table_schema, '.', table_name) AS full_table_name,
        privilege_type,
        is_grantable
    FROM {CATALOG}.information_schema.table_privileges
    WHERE table_schema != 'information_schema'
      AND table_name NOT LIKE 'mlflow_%'
""")

print(f"Found {privileges_df.count()} privilege grants")
privileges_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Build Rich Document Per Table (LLM-Enhanced Descriptions)

# COMMAND ----------

def build_table_document(table_row, columns, tags, col_tags, privileges):
    """Build a rich text document for a single table combining all metadata."""
    full_name = table_row["full_table_name"]

    # Header
    doc_parts = [
        f"# Table: {full_name}",
        f"Type: {table_row['table_type']}",
    ]

    if table_row.get("table_comment"):
        doc_parts.append(f"Description: {table_row['table_comment']}")

    # Columns section
    doc_parts.append("\n## Columns")
    for col in columns:
        col_line = f"- **{col['column_name']}** ({col['data_type']})"
        if col.get("column_comment"):
            col_line += f": {col['column_comment']}"
        if col.get("is_nullable") == "YES":
            col_line += " [nullable]"
        doc_parts.append(col_line)

    # Tags section
    if tags:
        doc_parts.append("\n## Tags")
        for tag in tags:
            doc_parts.append(f"- {tag['tag_name']}: {tag['tag_value']}")

    # Column tags
    if col_tags:
        doc_parts.append("\n## Column Tags")
        for ct in col_tags:
            doc_parts.append(f"- {ct['column_name']}.{ct['tag_name']}: {ct['tag_value']}")

    # Access / Privileges
    if privileges:
        doc_parts.append("\n## Access Privileges")
        for priv in privileges:
            grantable = " (WITH GRANT)" if priv.get("is_grantable") == "YES" else ""
            doc_parts.append(f"- {priv['grantee']}: {priv['privilege_type']}{grantable}")

    return "\n".join(doc_parts)

# COMMAND ----------

# Collect all data to driver for document assembly
tables_list = tables_df.collect()
columns_list = columns_df.collect()
tags_list = table_tags_df.collect() if has_tags else []
col_tags_list = column_tags_df.collect() if has_col_tags else []
privs_list = privileges_df.collect()

documents = []
for table in tables_list:
    full_name = table["full_table_name"]

    # Filter related metadata
    t_columns = [c.asDict() for c in columns_list if c["full_table_name"] == full_name]
    t_tags = [t.asDict() for t in tags_list if t["full_table_name"] == full_name]
    t_col_tags = [t.asDict() for t in col_tags_list if t["full_table_name"] == full_name]
    t_privs = [p.asDict() for p in privs_list if p["full_table_name"] == full_name]

    doc_text = build_table_document(table.asDict(), t_columns, t_tags, t_col_tags, t_privs)

    documents.append({
        "doc_id": full_name,
        "table_name": full_name,
        "catalog": table["table_catalog"],
        "schema": table["table_schema"],
        "table_type": table["table_type"],
        "content": doc_text,
        "column_names": ", ".join([c["column_name"] for c in t_columns]),
        "tag_summary": ", ".join([f"{t['tag_name']}={t['tag_value']}" for t in t_tags]) if t_tags else "",
        "acl_summary": ", ".join(set([p["grantee"] for p in t_privs])) if t_privs else "",
    })

print(f"Built {len(documents)} table documents")
for d in documents:
    print(f"\n{'='*60}")
    print(d["content"][:500])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Use LLM to Generate Rich NL Descriptions

# COMMAND ----------

import requests
import os

def generate_nl_description(doc_content):
    """Use Foundation Model API to generate a human-friendly description of the table."""
    token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    host = spark.conf.get("spark.databricks.workspaceUrl")

    response = requests.post(
        f"https://{host}/serving-endpoints/databricks-claude-sonnet-4/invocations",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={
            "messages": [
                {
                    "role": "system",
                    "content": "You are a data catalog assistant. Given table metadata, write a concise natural language description (2-4 sentences) that explains: what this table contains, what business domain it serves, key columns and their purpose, and how it might relate to other tables. Be specific and practical."
                },
                {
                    "role": "user",
                    "content": f"Generate a description for this table:\n\n{doc_content}"
                }
            ],
            "max_tokens": 300
        }
    )

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        print(f"LLM call failed: {response.status_code} {response.text}")
        return ""

# COMMAND ----------

# Enrich each document with an LLM-generated NL description
for doc in documents:
    nl_desc = generate_nl_description(doc["content"])
    doc["nl_description"] = nl_desc
    # Prepend the NL description to the content for richer vector embeddings
    doc["content"] = f"## AI-Generated Summary\n{nl_desc}\n\n{doc['content']}"
    print(f"  {doc['table_name']}: {nl_desc[:100]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Save as Delta Table & Create Vector Search Index

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType

schema = StructType([
    StructField("doc_id", StringType(), False),
    StructField("table_name", StringType(), False),
    StructField("catalog", StringType(), True),
    StructField("schema", StringType(), True),
    StructField("table_type", StringType(), True),
    StructField("content", StringType(), False),
    StructField("nl_description", StringType(), True),
    StructField("column_names", StringType(), True),
    StructField("tag_summary", StringType(), True),
    StructField("acl_summary", StringType(), True),
])

docs_df = spark.createDataFrame(documents, schema=schema)
docs_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(METADATA_TABLE)

# Enable Change Data Feed for sync index
spark.sql(f"ALTER TABLE {METADATA_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

print(f"Saved {docs_df.count()} documents to {METADATA_TABLE}")
docs_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Create Vector Search Endpoint & Index

# COMMAND ----------

vsc = VectorSearchClient()

# Create endpoint if it doesn't exist
try:
    vsc.get_endpoint(VS_ENDPOINT)
    print(f"Endpoint '{VS_ENDPOINT}' already exists")
except Exception:
    print(f"Creating endpoint '{VS_ENDPOINT}'...")
    vsc.create_endpoint(name=VS_ENDPOINT, endpoint_type="STANDARD")

# Wait for endpoint to be ready
import time
for i in range(60):
    ep = vsc.get_endpoint(VS_ENDPOINT)
    status = ep.get("endpoint_status", {}).get("state", "UNKNOWN")
    if status == "ONLINE":
        print(f"Endpoint is ONLINE")
        break
    print(f"  Waiting for endpoint... ({status})")
    time.sleep(10)

# COMMAND ----------

# Create the index (endpoint was freshly created above)
print(f"Creating Delta Sync index '{VS_INDEX}'...")
vsc.create_delta_sync_index(
    endpoint_name=VS_ENDPOINT,
    index_name=VS_INDEX,
    source_table_name=METADATA_TABLE,
    primary_key="doc_id",
    pipeline_type="TRIGGERED",
    embedding_source_column="content",
    embedding_model_endpoint_name="databricks-bge-large-en",
)
print("Index creation initiated. Waiting for sync...")

# COMMAND ----------

# Wait for index to be ready
for i in range(60):
    try:
        idx = vsc.get_index(endpoint_name=VS_ENDPOINT, index_name=VS_INDEX)
        status = idx.describe().get("status", {}).get("ready", False)
        if status:
            print("Index is READY!")
            break
        print(f"  Index syncing... ({i*10}s)")
    except Exception as e:
        print(f"  Waiting... {e}")
    time.sleep(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Test the Index with a Sample Query

# COMMAND ----------

idx = vsc.get_index(endpoint_name=VS_ENDPOINT, index_name=VS_INDEX)

results = idx.similarity_search(
    query_text="What tables have customer credit score information?",
    columns=["doc_id", "table_name", "nl_description", "content"],
    num_results=3,
)

for row in results.get("result", {}).get("data_array", []):
    print(f"\n{'='*60}")
    print(f"Table: {row[1]}")
    print(f"Description: {row[2][:200]}...")
    print(f"Score: {row[-1]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Done!
# MAGIC
# MAGIC The UC metadata has been extracted, enriched with AI-generated descriptions, and indexed
# MAGIC into Vector Search. The Databricks App will query this index to answer natural language
# MAGIC questions about your data catalog.
# MAGIC
# MAGIC **Index details:**
# MAGIC - Endpoint: `uc-discovery-vs-endpoint`
# MAGIC - Index: `experian_agent_demo_catalog.experian_agent_demo.uc_metadata_vs_index`
# MAGIC - Source table: `experian_agent_demo_catalog.experian_agent_demo.uc_metadata_documents`
