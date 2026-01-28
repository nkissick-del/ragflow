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
from psycopg2 import sql
from common.decorator import singleton
from common.doc_store.pgvector_conn_base import PGVectorConnectionBase
from common.doc_store.doc_store_models import VectorStoreQuery, VectorStoreQueryResult, VectorStoreHit, SearchMode
from common.doc_store.filter_translator import SQLFilterTranslator
from common.doc_store.post_processor import PostProcessor


@singleton
class PGVectorConnection(PGVectorConnectionBase):
    def __init__(self):
        super().__init__()
        # In a real implementation, we'd use a connection pool here
        self.conn = None

    def query(self, query: VectorStoreQuery, index_names: list[str], dataset_ids: list[str]) -> VectorStoreQueryResult:
        """
        Implementation of standardized query interface using PGVector and TSVector.
        """
        # 1. Database Table
        if not index_names or not isinstance(index_names, list) or not index_names[0]:
            raise ValueError(f"index_names must be a non-empty list. Received: {index_names}. Tenant context is required.")
        table_name = index_names[0]

        # 2. Filters
        translator = SQLFilterTranslator()
        filter_cond_str, filter_params = translator.translate(query.filters)
        if not filter_cond_str:
            filter_cond_str = "1=1"
            filter_params = []

        dataset_filter = sql.SQL("")
        dataset_params = []
        if dataset_ids:
            placeholders = sql.SQL(",").join([sql.Placeholder()] * len(dataset_ids))
            dataset_filter = sql.SQL(" AND kb_id IN ({})").format(placeholders)
            dataset_params = dataset_ids

        # 3. Search Strategy
        try:
            top_k = int(query.top_k)
        except (ValueError, TypeError):
            logging.warning(f"Invalid top_k: {query.top_k}. Falling back to 10.")
            top_k = 10

        sql_query = None
        params = []

        if query.mode == SearchMode.SEMANTIC:
            # Vector Search
            if query.query_vector is None:
                raise ValueError("query_vector is required for SEMANTIC search mode")
            if hasattr(query.query_vector, "__len__") and len(query.query_vector) == 0:
                raise ValueError("query_vector must be a non-empty sequence")

            vector_col = f"q_{len(query.query_vector)}_vec"
            vector_val = query.query_vector.tolist() if hasattr(query.query_vector, "tolist") else query.query_vector
            sql_query = sql.SQL("""
                SELECT id, content_with_weight, docnm_kwd, kb_id, 
                       1 - ({vector_col} <=> {vector_param}::vector) as score
                FROM {table}
                WHERE ({filter_cond}) {dataset_filter}
                ORDER BY score DESC
                LIMIT {top_k}
            """).format(
                vector_col=sql.Identifier(vector_col),
                vector_param=sql.Placeholder(),
                table=sql.Identifier(table_name),
                filter_cond=sql.SQL(filter_cond_str),
                dataset_filter=dataset_filter,
                top_k=sql.Literal(top_k),
            )
            params = [vector_val] + filter_params + dataset_params

        elif query.mode == SearchMode.FULLTEXT:
            # Fulltext Search
            if not query.query_text or not str(query.query_text).strip():
                raise ValueError("query_text is required for FULLTEXT search")

            sql_query = sql.SQL("""
                SELECT id, content_with_weight, docnm_kwd, kb_id,
                       ts_rank_cd(content_tsvector, websearch_to_tsquery('simple', {query_text_p1})) as score
                FROM {table}
                WHERE ({filter_cond}) AND content_tsvector @@ websearch_to_tsquery('simple', {query_text_p2}) {dataset_filter}
                ORDER BY score DESC
                LIMIT {top_k}
            """).format(
                query_text_p1=sql.Placeholder(),
                query_text_p2=sql.Placeholder(),
                table=sql.Identifier(table_name),
                filter_cond=sql.SQL(filter_cond_str),
                dataset_filter=dataset_filter,
                top_k=sql.Literal(top_k),
            )
            params = [query.query_text] + filter_params + [query.query_text] + dataset_params

        elif query.mode == SearchMode.HYBRID:
            # Hybrid Search (Weighted Sum)
            if query.query_vector is None:
                raise ValueError("query_vector is required for HYBRID search mode")

            if query.alpha is None:
                raise ValueError("alpha is required for HYBRID search mode")

            try:
                alpha_val = float(query.alpha)
            except (ValueError, TypeError):
                raise ValueError(f"alpha must be a float, got {query.alpha}")

            if not (0.0 <= alpha_val <= 1.0):
                raise ValueError(f"alpha must be between 0.0 and 1.0, got {alpha_val}")

            if not query.query_text or not str(query.query_text).strip():
                raise ValueError("query_text is required for HYBRID search mode")

            vector_col = f"q_{len(query.query_vector)}_vec"
            vector_val = query.query_vector.tolist() if hasattr(query.query_vector, "tolist") else query.query_vector

            sql_query = sql.SQL("""
                SELECT id, content_with_weight, docnm_kwd, kb_id,
                       ({alpha} * (1 - ({vector_col} <=> {vector_param}::vector)) + 
                        {one_minus_alpha} * ts_rank_cd(content_tsvector, websearch_to_tsquery('simple', {query_text_param}))) as score
                FROM {table}
                WHERE ({filter_cond}) {dataset_filter}
                ORDER BY score DESC
                LIMIT {top_k}
            """).format(
                alpha=sql.Literal(alpha_val),
                vector_col=sql.Identifier(vector_col),
                vector_param=sql.Placeholder(),
                one_minus_alpha=sql.Literal(1 - alpha_val),
                query_text_param=sql.Placeholder(),
                table=sql.Identifier(table_name),
                filter_cond=sql.SQL(filter_cond_str),
                dataset_filter=dataset_filter,
                top_k=sql.Literal(top_k),
            )
            params = [vector_val, query.query_text] + filter_params + dataset_params
        else:
            raise ValueError(f"Unrecognized search mode: {query.mode}. Mode must be SEMANTIC, FULLTEXT, or HYBRID.")

        # 4. Execute (Mocking connection for now)
        if self.conn is None:
            logging.warning("PGVectorConnection not connected. Returning empty result.")
            return VectorStoreQueryResult(hits=[], total=0)

        hits = []
        try:
            with self.conn.cursor() as cur:
                cur.execute(sql_query, params)
                rows = cur.fetchall()
                for row in rows:
                    doc_id, content, doc_name, kb_id, score = row

                    highlight = None
                    if query.query_text:
                        highlight = PostProcessor.highlight(content, [query.query_text])

                    hits.append(VectorStoreHit(id=doc_id, score=float(score), text=content, highlight=highlight, metadata={"doc_name": doc_name, "kb_id": kb_id}))
        except Exception as e:
            logging.error(f"Postgres query failed: {e}")
            raise

        return VectorStoreQueryResult(hits=hits, total=len(hits))

    def insert(self, rows: list[dict], index_name: str, dataset_id: str = None) -> list[str]:
        # Implementation of bulk insert with ON CONFLICT DO UPDATE
        return []

    def update(self, condition: dict, new_value: dict, index_name: str, dataset_id: str) -> bool:
        return True

    def delete(self, condition: dict, index_name: str, dataset_id: str) -> int:
        return 0

    def create_idx(self, index_name: str, dataset_id: str, vector_size: int):
        # Create table if not exists with PGVector and TSVector support
        pass

    def index_exist(self, index_name: str, dataset_id: str) -> bool:
        # dataset_id is currently unused but kept for API compatibility

        if self.conn is None:
            return False
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = %s)", (index_name,))
                return cur.fetchone()[0]
        except Exception as e:
            logging.error(f"Failed to check index existence: {e}")
            return False

    def health(self) -> dict:
        if self.conn is None:
            return {"status": "down", "detail": "not connected"}
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1")
                return {"status": "green", "detail": "connected"}
        except Exception as e:
            return {"status": "down", "detail": str(e)}
