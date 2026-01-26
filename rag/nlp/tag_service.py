#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
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

from common.doc_store.doc_store_base import OrderByExpr
from common.constants import TAG_FLD


SCORE_SCALE = 0.1  # Scales raw tag scores to final score range
EPSILON = 1e-6  # Prevents division-by-zero
DEFAULT_TAG_FREQ = 0.0001  # Fallback tag frequency used in smoothing
DEFAULT_S = 1000  # Smoothing constant in the smoothing formula


def index_name(uid):
    return f"ragflow_{uid}"


class TagService:
    def __init__(self, dataStore, qryr):
        self.dataStore = dataStore
        self.qryr = qryr

    def all_tags(self, tenant_id: str, kb_ids: list[str]):
        if not kb_ids:
            return []
        if not self.dataStore.index_exist(index_name(tenant_id), kb_ids[0]):
            return []
        res = self.dataStore.search([], [], {}, [], OrderByExpr(), 0, 0, index_name(tenant_id), kb_ids, ["tag_kwd"])
        return self.dataStore.get_aggregation(res, "tag_kwd")

    def all_tags_in_portion(self, tenant_id: str, kb_ids: list[str], S=DEFAULT_S):
        if not kb_ids:
            return {}
        if not self.dataStore.index_exist(index_name(tenant_id), kb_ids[0]):
            return {}
        res = self.dataStore.search([], [], {}, [], OrderByExpr(), 0, 0, index_name(tenant_id), kb_ids, ["tag_kwd"])
        res = self.dataStore.get_aggregation(res, "tag_kwd")
        total = sum(c for _, c in res)
        return {t: (c + 1) / (total + S) for t, c in res}

    def tag_content(self, tenant_id: str, kb_ids: list[str], doc, all_tags, topn_tags=3, keywords_topn=30, S=DEFAULT_S):
        """
        Calculate tags for a document based on content matching and tag frequency.
        Score = SCORE_SCALE * (match_count + 1) / (total_matches + S) / max(EPSILON, global_tag_freq)
        """
        idx_nm = index_name(tenant_id)
        match_txt = self.qryr.paragraph(doc["title_tks"] + " " + doc["content_ltks"], doc.get("important_kwd", []), keywords_topn)
        res = self.dataStore.search([], [], {}, [match_txt], OrderByExpr(), 0, 0, idx_nm, kb_ids, ["tag_kwd"])
        aggs = self.dataStore.get_aggregation(res, "tag_kwd")
        if not aggs:
            return False
        cnt = sum(c for _, c in aggs)
        tag_fea = self._compute_tag_scores(aggs, all_tags, cnt, S, topn_tags)
        doc[TAG_FLD] = {a.replace(".", "_"): c for a, c in tag_fea if c >= 0.001}
        return True

    def _compute_tag_scores(self, aggs, all_tags, cnt, S, topn_tags):
        return sorted([(a, SCORE_SCALE * (c + 1) / (cnt + S) / max(EPSILON, all_tags.get(a, DEFAULT_TAG_FREQ))) for a, c in aggs], key=lambda x: x[1] * -1)[:topn_tags]

    def tag_query(self, question: str, tenant_ids: str | list[str], kb_ids: list[str], all_tags, topn_tags=3, S=DEFAULT_S):
        if isinstance(tenant_ids, str):
            idx_nms = index_name(tenant_ids)
        else:
            idx_nms = [index_name(tid) for tid in tenant_ids]
        match_txt, _ = self.qryr.question(question, min_match=0.0)
        res = self.dataStore.search([], [], {}, [match_txt], OrderByExpr(), 0, 0, idx_nms, kb_ids, ["tag_kwd"])
        aggs = self.dataStore.get_aggregation(res, "tag_kwd")
        if not aggs:
            return {}
        cnt = sum(c for _, c in aggs)
        tag_fea = self._compute_tag_scores(aggs, all_tags, cnt, S, topn_tags)
        return {a.replace(".", "_"): c for a, c in tag_fea if c >= 0.001}
