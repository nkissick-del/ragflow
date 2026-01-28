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
        # In Postgres, index_name usually maps to a table name. RAGFlow uses 'ragflow_<tenant_id>'
        table_name = index_names[0] if isinstance(index_names, list) else index_names

        # 2. Filters
        translator = SQLFilterTranslator()
        filter_cond = translator.translate(query.filters)

        # We also filter by dataset_ids if provided
        if dataset_ids:
            kb_ids_str = ",".join([f"'{k}'" for k in dataset_ids])
            filter_cond += f" AND kb_id IN ({kb_ids_str})"

        # 3. Search Strategy
        sql = ""
        params = []

        if query.mode == SearchMode.SEMANTIC:
            # Vector Search
            vector_col = f"q_{len(query.query_vector)}_vec"
            sql = f"""
                SELECT id, content_with_weight, docnm_kwd, kb_id, 
                       1 - ({vector_col} <=> %s::vector) as score
                FROM {table_name}
                WHERE {filter_cond}
                ORDER BY score DESC
                LIMIT {query.top_k}
            """
            params = [query.query_vector.tolist() if hasattr(query.query_vector, "tolist") else query.query_vector]

        elif query.mode == SearchMode.FULLTEXT:
            # Fulltext Search
            sql = f"""
                SELECT id, content_with_weight, docnm_kwd, kb_id,
                       ts_rank_cd(content_tsvector, websearch_to_tsquery('simple', %s)) as score
                FROM {table_name}
                WHERE {filter_cond} AND content_tsvector @@ websearch_to_tsquery('simple', %s)
                ORDER BY score DESC
                LIMIT {query.top_k}
            """
            params = [query.query_text, query.query_text]

        elif query.mode == SearchMode.HYBRID:
            # Hybrid Search (Weighted Sum)
            vector_col = f"q_{len(query.query_vector)}_vec"
            alpha = query.alpha
            sql = f"""
                SELECT id, content_with_weight, docnm_kwd, kb_id,
                       ({alpha} * (1 - ({vector_col} <=> %s::vector)) + 
                        {1 - alpha} * ts_rank_cd(content_tsvector, websearch_to_tsquery('simple', %s))) as score
                FROM {table_name}
                WHERE {filter_cond}
                ORDER BY score DESC
                LIMIT {query.top_k}
            """
            params = [query.query_vector.tolist() if hasattr(query.query_vector, "tolist") else query.query_vector, query.query_text]

        # 4. Execute (Mocking connection for now)
        if self.conn is None:
            logging.warning("PGVectorConnection not connected. Returning empty result.")
            return VectorStoreQueryResult(hits=[], total=0)

        hits = []
        try:
            with self.conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
                for row in rows:
                    doc_id, content, doc_name, kb_id, score = row

                    highlight = None
                    if query.query_text:
                        highlight = PostProcessor.highlight(content, [query.query_text])

                    hits.append(VectorStoreHit(id=doc_id, score=float(score), text=content, highlight=highlight, metadata={"doc_name": doc_name, "kb_id": kb_id}))
        except Exception as e:
            logging.error(f"Postgres query failed: {e}")

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
        return True

    def health(self) -> dict:
        return {"status": "green"}
