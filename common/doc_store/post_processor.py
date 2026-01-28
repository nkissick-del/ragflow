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
import re
from typing import List, Any
from rag.nlp import is_english


class PostProcessor:
    @staticmethod
    def highlight(text: str, keywords: List[str]) -> str:
        """
        Universal highligher that wraps keywords in <em> tags and returns snippet.
        Ported from es_conn_base.py and infinity_conn_base.py.
        """
        if not text or not keywords:
            return text

        # Clean up newlines for better snippet generation
        txt = re.sub(r"[\r\n]", " ", text, flags=re.IGNORECASE | re.MULTILINE)
        txt_list = []

        # Split into sentences
        sentences = re.split(r"[.?!;\n]", txt)

        for t in sentences:
            found = False
            if is_english([t]):
                for w in keywords:
                    # Case-insensitive replacement with boundary check for English
                    pattern = r"(^|[ .?/'\"\(\)!,:;-])(%s)([ .?/'\"\(\)!,:;-])" % re.escape(w)
                    t_new = re.sub(pattern, r"\1<em>\2</em>\3", t, flags=re.IGNORECASE | re.MULTILINE)
                    if t_new != t:
                        t = t_new
                        found = True
            else:
                # For non-English (e.g. Chinese), match substrings directly
                for w in sorted(keywords, key=len, reverse=True):
                    pattern = re.escape(w)
                    t_new = re.sub(pattern, f"<em>{w}</em>", t, flags=re.IGNORECASE | re.MULTILINE)
                    if t_new != t:
                        t = t_new
                        found = True

            if found:
                txt_list.append(t)

        if txt_list:
            return "...".join(txt_list)
        return text[:200] + "..."  # Fallback for no keywords found in snippet

    @staticmethod
    def normalize_scores(hits: List[Any], distance_type: str = "cosine") -> List[Any]:
        """
        Skeleton for score normalization.
        In the future, this will map distance metrics to a standard 0-1 relevance scale.
        """
        return hits
