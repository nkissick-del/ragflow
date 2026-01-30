#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import logging
import re
from typing import Union, List

from psycopg2 import sql

from common.decorator import singleton
from common.doc_store.pgvector_conn_base import PGVectorConnectionBase
from common.doc_store.doc_store_base import MatchExpr, MatchTextExpr, MatchDenseExpr, FusionExpr, OrderByExpr
from common.doc_store.doc_store_models import VectorStoreQuery, VectorStoreQueryResult, VectorStoreHit, SearchMode
from common.doc_store.filter_translator import SQLFilterTranslator
from common.doc_store.post_processor import PostProcessor


# Supported vector dimensions
SUPPORTED_VECTOR_DIMS = [256, 512, 768, 1024, 1536, 3072, 4096]

# Allowed columns for SQL validation
ALLOWED_COLUMNS = {
    "id",
    "kb_id",
    "doc_id",
    "docnm_kwd",
    "content_with_weight",
    "content_ltks",
    "content_sm_ltks",
    "title_tks",
    "title_sm_tks",
    "important_kwd",
    "important_tks",
    "position_int",
    "page_num_int",
    "top_int",
    "tag_feas",
    "tag_kwd",
    "knowledge_graph_kwd",
    "question_kwd",
    "question_tks",
    "image_id",
    "img_id",
    "available_int",
    "removed_kwd",
    "create_time",
    "create_timestamp",
    "update_time",
    "content_tsvector",
}
for dim in SUPPORTED_VECTOR_DIMS:
    ALLOWED_COLUMNS.add(f"q_{dim}_vec")

TABLE_NAME_REGEX = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Table creation SQL template
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS {table_name} (
    id VARCHAR(64) PRIMARY KEY,
    kb_id VARCHAR(64) NOT NULL,
    doc_id VARCHAR(64),
    docnm_kwd VARCHAR(1024),
    content_with_weight TEXT,
    content_ltks TEXT,
    content_sm_ltks TEXT,
    title_tks TEXT,
    title_sm_tks TEXT,
    important_kwd TEXT[],
    important_tks TEXT,
    position_int TEXT[],
    page_num_int INTEGER[],
    top_int INTEGER[],
    tag_feas JSONB DEFAULT '{{}}',
    tag_kwd TEXT[],
    knowledge_graph_kwd TEXT,
    question_kwd TEXT,
    question_tks TEXT,
    image_id VARCHAR(256),
    img_id VARCHAR(256),
    available_int INTEGER DEFAULT 1,
    removed_kwd VARCHAR(1),
    create_time BIGINT,
    create_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- Vector columns for different embedding dimensions
    q_256_vec VECTOR(256),
    q_512_vec VECTOR(512),
    q_768_vec VECTOR(768),
    q_1024_vec VECTOR(1024),
    q_1536_vec VECTOR(1536),
    q_3072_vec VECTOR(3072),
    q_4096_vec VECTOR(4096),
    -- Full-text search vector (generated)
    content_tsvector TSVECTOR
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_{table_name_safe}_kb_id ON {table_name} (kb_id);
CREATE INDEX IF NOT EXISTS idx_{table_name_safe}_doc_id ON {table_name} (doc_id);
CREATE INDEX IF NOT EXISTS idx_{table_name_safe}_available ON {table_name} (available_int);
CREATE INDEX IF NOT EXISTS idx_{table_name_safe}_tsvector ON {table_name} USING GIN (content_tsvector);
"""

# Trigger for updating tsvector
CREATE_TSVECTOR_TRIGGER_SQL = """
CREATE OR REPLACE FUNCTION update_content_tsvector_{table_name_safe}()
RETURNS TRIGGER AS $$
BEGIN
    NEW.content_tsvector := to_tsvector('simple', COALESCE(NEW.content_with_weight, '') || ' ' || COALESCE(NEW.title_tks, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_update_tsvector_{table_name_safe} ON {table_name};
CREATE TRIGGER trg_update_tsvector_{table_name_safe}
    BEFORE INSERT OR UPDATE ON {table_name}
    FOR EACH ROW
    EXECUTE FUNCTION update_content_tsvector_{table_name_safe}();
"""


@singleton
class PGVectorConnection(PGVectorConnectionBase):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger("ragflow.pgvector_conn")
        # Initialize connection pool
        from common.doc_store.pgvector_conn_pool import get_pgvector_conn

        self._pool = get_pgvector_conn()
        self.logger.info("PGVectorConnection initialized")

    def health(self) -> dict:
        """Return health status of the database."""
        try:
            with self._pool.cursor() as cur:
                cur.execute("SELECT 1")
                return {"status": "green", "type": "pgvector", "detail": "connected"}
        except Exception as e:
            return {"status": "red", "type": "pgvector", "detail": str(e)}

    def create_idx(self, index_name: str, dataset_id: str, vector_size: int, parser_id: str = None):
        """Create a table for storing document chunks with vector embeddings."""
        # Sanitize table name for use in identifiers
        table_name_safe = re.sub(r"[^a-zA-Z0-9_]", "_", index_name)

        try:
            with self._pool.cursor() as cur:
                # Create table
                create_sql = CREATE_TABLE_SQL.format(table_name=sql.Identifier(index_name).as_string(cur.connection), table_name_safe=table_name_safe)
                cur.execute(create_sql)

                # Create vector indexes for all supported dimensions
                for dim in SUPPORTED_VECTOR_DIMS:
                    vector_col = f"q_{dim}_vec"
                    index_name_safe = f"idx_{table_name_safe}_{vector_col}"
                    # Use IVFFlat for very high dimensions to save disk, or stick to HNSW
                    # The user requested HNSW for all but mentioned IVFFlat as an option for high dims.
                    # "for very high dims (3072, 4096) make this conditional or document/choose IVFFlat instead"
                    method = "hnsw"
                    ops = "vector_cosine_ops"

                    idx_sql = sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} USING {} ({} {})").format(
                        sql.Identifier(index_name_safe), sql.Identifier(index_name), sql.SQL(method), sql.Identifier(vector_col), sql.SQL(ops)
                    )
                    cur.execute(idx_sql)

                # Create tsvector trigger
                trigger_sql = CREATE_TSVECTOR_TRIGGER_SQL.format(table_name=sql.Identifier(index_name).as_string(cur.connection), table_name_safe=table_name_safe)
                cur.execute(trigger_sql)

            self.logger.info(f"Created pgvector table: {index_name}")
            return True
        except Exception as e:
            self.logger.exception(f"Failed to create pgvector table {index_name}: {e}")
            return False

    def delete_idx(self, index_name: str, dataset_id: str):
        """Delete a table (only if dataset_id is empty - full tenant deletion)."""
        if dataset_id:
            # Don't drop table if only deleting a dataset within tenant
            return
        try:
            with self._pool.cursor() as cur:
                cur.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(index_name)))
            self.logger.info(f"Dropped pgvector table: {index_name}")
        except Exception as e:
            self.logger.exception(f"Failed to drop pgvector table {index_name}: {e}")

    def index_exist(self, index_name: str, dataset_id: str = None) -> bool:
        """Check if table exists."""
        try:
            with self._pool.cursor() as cur:
                cur.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = %s)", (index_name,))
                return cur.fetchone()[0]
        except Exception as e:
            self.logger.error(f"Failed to check table existence: {e}")
            return False

    def query(self, query: VectorStoreQuery, index_names: list[str], dataset_ids: list[str]) -> VectorStoreQueryResult:
        """Standardized query interface using VectorStoreQuery."""
        if not index_names or not index_names[0]:
            raise ValueError(f"index_names must be a non-empty list. Received: {index_names}")

        table_name = index_names[0]
        if not TABLE_NAME_REGEX.match(table_name):
            raise ValueError(f"Invalid table name: {table_name}")

        # Build filter conditions
        translator = SQLFilterTranslator()
        filter_cond_str, filter_params = translator.translate(query.filters)
        if not filter_cond_str:
            filter_cond_str = "1=1"
            filter_params = []

        # Dataset filter
        dataset_filter = sql.SQL("")
        dataset_params = []
        if dataset_ids:
            placeholders = sql.SQL(",").join([sql.Placeholder()] * len(dataset_ids))
            dataset_filter = sql.SQL(" AND kb_id IN ({})").format(placeholders)
            dataset_params = dataset_ids

        top_k = int(query.top_k) if query.top_k else 10

        sql_query = None
        params = []

        try:
            with self._pool.cursor(commit=False) as cur:
                if query.mode == SearchMode.SEMANTIC:
                    if query.query_vector is None:
                        raise ValueError("query_vector is required for SEMANTIC search mode")

                    vector_dim = len(query.query_vector)
                    if vector_dim not in SUPPORTED_VECTOR_DIMS:
                        raise ValueError(f"Unsupported vector dimension: {vector_dim}. Supported: {SUPPORTED_VECTOR_DIMS}")
                    vector_col = f"q_{vector_dim}_vec"
                    vector_val = list(query.query_vector) if hasattr(query.query_vector, "tolist") else query.query_vector

                    sql_query = sql.SQL("""
                        SELECT id, content_with_weight, docnm_kwd, kb_id,
                               1 - ({} <=> %s::vector) as score
                        FROM {}
                        WHERE ({}) {}
                        ORDER BY score DESC
                        LIMIT %s
                    """).format(sql.Identifier(vector_col), sql.Identifier(table_name), sql.SQL(filter_cond_str), dataset_filter)
                    params = [vector_val] + filter_params + dataset_params + [top_k]

                elif query.mode == SearchMode.FULLTEXT:
                    if not query.query_text:
                        raise ValueError("query_text is required for FULLTEXT search")

                    sql_query = sql.SQL("""
                        SELECT id, content_with_weight, docnm_kwd, kb_id,
                               ts_rank_cd(content_tsvector, websearch_to_tsquery('simple', %s)) as score
                        FROM {}
                        WHERE ({}) AND content_tsvector @@ websearch_to_tsquery('simple', %s) {}
                        ORDER BY score DESC
                        LIMIT %s
                    """).format(sql.Identifier(table_name), sql.SQL(filter_cond_str), dataset_filter)
                    params = [query.query_text] + filter_params + [query.query_text] + dataset_params + [top_k]

                elif query.mode == SearchMode.HYBRID:
                    if query.query_vector is None:
                        raise ValueError("query_vector is required for HYBRID search mode")
                    if not query.query_text:
                        raise ValueError("query_text is required for HYBRID search mode")

                    alpha = float(query.alpha) if query.alpha else 0.5
                    vector_dim = len(query.query_vector)
                    if vector_dim not in SUPPORTED_VECTOR_DIMS:
                        raise ValueError(f"Unsupported vector dimension: {vector_dim}. Supported: {SUPPORTED_VECTOR_DIMS}")
                    vector_col = f"q_{vector_dim}_vec"
                    vector_val = list(query.query_vector) if hasattr(query.query_vector, "tolist") else query.query_vector

                    sql_query = sql.SQL("""
                        SELECT id, content_with_weight, docnm_kwd, kb_id,
                               (%s * (1 - ({} <=> %s::vector)) + 
                                %s * COALESCE(ts_rank_cd(content_tsvector, websearch_to_tsquery('simple', %s)), 0)) as score
                        FROM {}
                        WHERE ({}) {}
                        ORDER BY score DESC
                        LIMIT %s
                    """).format(sql.Identifier(vector_col), sql.Identifier(table_name), sql.SQL(filter_cond_str), dataset_filter)
                    params = [alpha, vector_val, 1.0 - alpha, query.query_text] + filter_params + dataset_params + [top_k]
                else:
                    raise ValueError(f"Unrecognized search mode: {query.mode}")

                cur.execute(sql_query, params)
                rows = cur.fetchall()

                hits = []
                for row in rows:
                    doc_id, content, doc_name, kb_id, score = row
                    highlight = None
                    if query.query_text:
                        highlight = PostProcessor.highlight(content or "", [query.query_text])
                    hits.append(VectorStoreHit(id=doc_id, score=float(score) if score else 0.0, text=content or "", highlight=highlight, metadata={"doc_name": doc_name, "kb_id": kb_id}))

                return VectorStoreQueryResult(hits=hits, total=len(hits))

        except Exception as e:
            self.logger.error(f"PGVector query failed: {e}")
            raise

    def search(
        self,
        select_fields: list[str],
        highlight_fields: list[str],
        condition: dict,
        match_expressions: list[MatchExpr],
        order_by: OrderByExpr,
        offset: int,
        limit: int,
        index_names: Union[str, List[str]],
        dataset_ids: list[str],
        agg_fields: list[str] | None = None,
        rank_feature: dict | None = None,
    ) -> dict:
        """Full search implementation compatible with ES interface."""
        if isinstance(index_names, str):
            index_names = index_names.split(",")
        if not index_names:
            raise ValueError("index_names required")

        table_name = index_names[0]
        if not TABLE_NAME_REGEX.match(table_name):
            raise ValueError(f"Invalid table name: {table_name}")

        # Build WHERE clause
        where_parts = []
        where_params = []

        # Add kb_id filter
        if dataset_ids:
            placeholders = sql.SQL(",").join([sql.Placeholder()] * len(dataset_ids))
            where_parts.append(sql.SQL("kb_id IN ({})").format(placeholders))
            where_params.extend(dataset_ids)

        # Add other conditions
        for k, v in condition.items():
            if k == "kb_id":
                continue  # Already handled
            if k not in ALLOWED_COLUMNS:
                self.logger.warning(f"Skipping forbidden column in condition: {k}")
                continue

            if k == "available_int":
                if v == 0:
                    where_parts.append(sql.SQL("available_int < 1"))
                else:
                    where_parts.append(sql.SQL("available_int >= 1"))
                continue
            if v is None:
                continue
            if isinstance(v, list):
                if not v:
                    continue
                placeholders = sql.SQL(",").join([sql.Placeholder()] * len(v))
                where_parts.append(sql.SQL("{} IN ({})").format(sql.Identifier(k), placeholders))
                where_params.extend(v)
            elif isinstance(v, (str, int)):
                where_parts.append(sql.SQL("{} = %s").format(sql.Identifier(k)))
                where_params.append(v)

        where_clause = sql.SQL(" AND ").join(where_parts) if where_parts else sql.SQL("1=1")

        # Handle match expressions
        vector_parts = []
        text_parts = []
        vector_weight = 0.5

        for m in match_expressions:
            if isinstance(m, FusionExpr) and m.method == "weighted_sum" and m.fusion_params:
                weights = m.fusion_params.get("weights", "0.5,0.5")
                try:
                    vector_weight = float(weights.split(",")[1])
                except (ValueError, IndexError):
                    vector_weight = 0.5
            elif isinstance(m, MatchDenseExpr):
                vector_parts.append(m)
            elif isinstance(m, MatchTextExpr):
                text_parts.append(m)

        # Build score expression
        score_expr = sql.SQL("0")
        score_params = []
        match_where_clause = sql.SQL("")
        match_where_params = []

        if vector_parts and text_parts:
            # Hybrid
            m = vector_parts[0]
            if m.vector_column_name not in ALLOWED_COLUMNS:
                raise ValueError(f"Invalid vector column: {m.vector_column_name}")
            vector_val = list(m.embedding_data) if hasattr(m.embedding_data, "tolist") else m.embedding_data
            tm = text_parts[0]
            score_expr = sql.SQL("({} * (1 - ({} <=> %s::vector)) + {} * COALESCE(ts_rank_cd(content_tsvector, websearch_to_tsquery('simple', %s)), 0))").format(
                sql.Literal(vector_weight), sql.Identifier(m.vector_column_name), sql.Literal(1.0 - vector_weight)
            )
            score_params = [vector_val, tm.matching_text]
        elif vector_parts:
            m = vector_parts[0]
            if m.vector_column_name not in ALLOWED_COLUMNS:
                raise ValueError(f"Invalid vector column: {m.vector_column_name}")
            vector_val = list(m.embedding_data) if hasattr(m.embedding_data, "tolist") else m.embedding_data
            score_expr = sql.SQL("(1 - ({} <=> %s::vector))").format(sql.Identifier(m.vector_column_name))
            score_params = [vector_val]
        elif text_parts:
            tm = text_parts[0]
            score_expr = sql.SQL("ts_rank_cd(content_tsvector, websearch_to_tsquery('simple', %s))")
            score_params = [tm.matching_text]
            match_where_clause = sql.SQL(" AND content_tsvector @@ websearch_to_tsquery('simple', %s)")
            match_where_params = [tm.matching_text]

        # Build SELECT
        if not select_fields:
            select_clause = sql.SQL("*")
        else:
            # Validate select fields
            valid_select = []
            for f in select_fields:
                if f in ALLOWED_COLUMNS or f == "*":
                    valid_select.append(sql.Identifier(f) if f != "*" else sql.SQL("*"))
                else:
                    self.logger.warning(f"Skipping forbidden column in select: {f}")

            if not valid_select:
                self.logger.error(f"No valid fields found in select_fields: {select_fields}")
                raise ValueError(f"No valid fields found in select_fields: {select_fields}")

            select_clause = sql.SQL(", ").join(valid_select)

        # Build ORDER BY
        order_clause = sql.SQL("score DESC")
        if order_by and order_by.fields_prop:
            order_parts = []
            for field, direction in order_by.fields_prop:
                if field not in ALLOWED_COLUMNS and field != "score":
                    self.logger.warning(f"Skipping forbidden column in order by: {field}")
                    continue
                dir_str = "ASC" if direction == 0 else "DESC"
                order_parts.append(sql.SQL("{} {}").format(sql.Identifier(field) if field != "score" else sql.SQL("score"), sql.SQL(dir_str)))
            if order_parts:
                order_clause = sql.SQL(", ").join(order_parts)

        query_sql = sql.SQL("""
            SELECT {}, {} as score
            FROM {}
            WHERE ({}) {}
            ORDER BY {}
            OFFSET %s LIMIT %s
        """).format(select_clause, score_expr, sql.Identifier(table_name), where_clause, match_where_clause, order_clause)
        params = score_params + where_params + match_where_params + [offset, limit]

        try:
            with self._pool.cursor(commit=False) as cur:
                cur.execute(query_sql, params)
                rows = cur.fetchall()
                colnames = [desc[0] for desc in cur.description]

                hits = []
                for row in rows:
                    doc = dict(zip(colnames, row))
                    doc["_score"] = doc.pop("score", 0)
                    doc["_id"] = doc.get("id")
                    hits.append({"_id": doc.get("id"), "_score": doc.get("_score"), "_source": doc})

                # Get total count
                count_sql = sql.SQL("SELECT COUNT(*) FROM {} WHERE ({}) {}").format(sql.Identifier(table_name), where_clause, match_where_clause)
                count_params = where_params + match_where_params
                cur.execute(count_sql, count_params)
                total = cur.fetchone()[0]

                return {"hits": {"hits": hits, "total": {"value": total}}}
        except Exception as e:
            self.logger.error(f"PGVector search failed: {e}")
            raise

    def get(self, doc_id: str, index_name: str, dataset_ids: list[str]) -> dict | None:
        """Get single document by ID."""
        try:
            with self._pool.cursor(commit=False) as cur:
                cur.execute(sql.SQL("SELECT * FROM {} WHERE id = %s").format(sql.Identifier(index_name)), (doc_id,))
                row = cur.fetchone()
                if not row:
                    return None
                colnames = [desc[0] for desc in cur.description]
                doc = dict(zip(colnames, row))
                doc["id"] = doc_id
                return doc
        except Exception as e:
            self.logger.error(f"PGVector get failed: {e}")
            return None

    def insert(self, rows: list[dict], index_name: str, dataset_id: str = None) -> list[str]:
        """Bulk insert documents with ON CONFLICT DO UPDATE."""
        if not rows:
            return []

        if not TABLE_NAME_REGEX.match(index_name):
            raise ValueError(f"Invalid table name: {index_name}")

        errors = []
        try:
            with self._pool.cursor() as cur:
                for doc in rows:
                    doc_copy = doc.copy()
                    doc_id = doc_copy.pop("id", None)
                    if not doc_id:
                        errors.append("Document missing 'id' field")
                        continue

                    if dataset_id:
                        doc_copy["kb_id"] = dataset_id

                    # Handle vector fields
                    for dim in SUPPORTED_VECTOR_DIMS:
                        vec_field = f"q_{dim}_vec"
                        if vec_field in doc_copy:
                            vec_val = doc_copy[vec_field]
                            if hasattr(vec_val, "tolist"):
                                doc_copy[vec_field] = vec_val.tolist()

                    # Build insert statement
                    valid_cols = []
                    valid_vals = []
                    for col, val in doc_copy.items():
                        if col in ALLOWED_COLUMNS:
                            valid_cols.append(col)
                            valid_vals.append(val)
                        else:
                            self.logger.warning(f"Skipping forbidden column in insert: {col}")

                    columns = [sql.Identifier("id")] + [sql.Identifier(col) for col in valid_cols]
                    values = [doc_id] + valid_vals

                    placeholders = sql.SQL(", ").join([sql.Placeholder()] * len(values))
                    col_sql = sql.SQL(", ").join(columns)

                    insert_sql = sql.SQL("""
                        INSERT INTO {} ({})
                        VALUES ({})
                        ON CONFLICT (id) DO NOTHING
                    """).format(sql.Identifier(index_name), col_sql, placeholders)

                    try:
                        cur.execute(insert_sql, values)
                    except Exception as e:
                        errors.append(f"{doc_id}: {str(e)}")

        except Exception as e:
            errors.append(str(e))
            self.logger.exception(f"PGVector insert failed: {e}")

        return errors

    def update(self, condition: dict, new_value: dict, index_name: str, dataset_id: str) -> bool:
        """Update documents matching condition."""
        if not TABLE_NAME_REGEX.match(index_name):
            raise ValueError(f"Invalid table name: {index_name}")

        try:
            # Build SET clause
            set_parts = []
            set_params = []

            for k, v in new_value.items():
                if k in ("id", "remove", "add"):
                    continue
                if k not in ALLOWED_COLUMNS:
                    self.logger.warning(f"Skipping forbidden column in update SET: {k}")
                    continue
                set_parts.append(sql.SQL("{} = %s").format(sql.Identifier(k)))
                set_params.append(v)

            if not set_parts:
                return True

            # Build WHERE clause
            where_parts = [sql.SQL("kb_id = %s")]
            where_params = [dataset_id]

            # Handle specific ID update
            if "id" in condition and isinstance(condition["id"], str):
                where_parts.append(sql.SQL("id = %s"))
                where_params.append(condition["id"])
            else:
                for k, v in condition.items():
                    if k == "id":
                        continue
                    if k not in ALLOWED_COLUMNS:
                        self.logger.warning(f"Skipping forbidden column in update WHERE: {k}")
                        continue
                    if isinstance(v, list):
                        if not v:
                            continue
                        placeholders = sql.SQL(",").join([sql.Placeholder()] * len(v))
                        where_parts.append(sql.SQL("{} IN ({})").format(sql.Identifier(k), placeholders))
                        where_params.extend(v)
                    elif isinstance(v, (str, int)):
                        where_parts.append(sql.SQL("{} = %s").format(sql.Identifier(k)))
                        where_params.append(v)

            set_clause = sql.SQL(", ").join(set_parts)
            where_clause = sql.SQL(" AND ").join(where_parts)

            update_sql = sql.SQL("UPDATE {} SET {} WHERE {}").format(sql.Identifier(index_name), set_clause, where_clause)
            params = set_params + where_params

            with self._pool.cursor() as cur:
                cur.execute(update_sql, params)

            return True
        except Exception as e:
            self.logger.exception(f"PGVector update failed: {e}")
            return False

    def delete(self, condition: dict, index_name: str, dataset_id: str) -> int:
        """Delete documents matching condition."""
        if not TABLE_NAME_REGEX.match(index_name):
            raise ValueError(f"Invalid table name: {index_name}")

        try:
            where_parts = [sql.SQL("kb_id = %s")]
            params = [dataset_id]

            if "id" in condition:
                ids = condition["id"]
                if not isinstance(ids, list):
                    ids = [ids]
                if ids:
                    placeholders = sql.SQL(",").join([sql.Placeholder()] * len(ids))
                    where_parts.append(sql.SQL("id IN ({})").format(placeholders))
                    params.extend(ids)

            for k, v in condition.items():
                if k == "id":
                    continue
                if k == "exists":
                    if v in ALLOWED_COLUMNS:
                        where_parts.append(sql.SQL("{} IS NOT NULL").format(sql.Identifier(v)))
                    else:
                        self.logger.warning(f"Skipping forbidden column in delete exists: {v}")
                    continue

                if k not in ALLOWED_COLUMNS:
                    self.logger.warning(f"Skipping forbidden column in delete condition: {k}")
                    continue

                if isinstance(v, list):
                    if not v:
                        continue
                    placeholders = sql.SQL(",").join([sql.Placeholder()] * len(v))
                    where_parts.append(sql.SQL("{} IN ({})").format(sql.Identifier(k), placeholders))
                    params.extend(v)
                elif isinstance(v, (str, int)):
                    where_parts.append(sql.SQL("{} = %s").format(sql.Identifier(k)))
                    params.append(v)

            where_clause = sql.SQL(" AND ").join(where_parts)
            delete_sql = sql.SQL("DELETE FROM {} WHERE {}").format(sql.Identifier(index_name), where_clause)

            with self._pool.cursor() as cur:
                cur.execute(delete_sql, params)
                return cur.rowcount

        except Exception as e:
            self.logger.exception(f"PGVector delete failed: {e}")
            return 0

    def sql(self, sql_str: str, fetch_size: int, format: str):
        """Execute raw SQL (for text-to-sql)."""
        try:
            with self._pool.cursor(commit=False) as cur:
                cur.execute(sql_str)
                if format == "json":
                    colnames = [desc[0] for desc in cur.description]
                    rows = cur.fetchmany(fetch_size)
                    return {"columns": colnames, "rows": [dict(zip(colnames, row)) for row in rows]}
                else:
                    return cur.fetchmany(fetch_size)
        except Exception as e:
            self.logger.error(f"PGVector SQL execution failed: {e}")
            raise Exception(f"SQL error: {e}\n\nSQL: {sql_str}")
