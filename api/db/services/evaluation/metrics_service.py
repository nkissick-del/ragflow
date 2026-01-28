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
from typing import List, Dict, Any, Optional

import json_repair
from api.db.services.llm_service import LLMBundle
from api.db.services.tenant_llm_service import TenantLLMService
from common.constants import LLMType
from rag.prompts.generator import PROMPT_JINJA_ENV, message_fit_in
from rag.prompts.template import load_prompt


class EvaluationMetricsService:
    @classmethod
    def compute_metrics(
        cls, question: str, generated_answer: str, reference_answer: Optional[str], retrieved_chunks: List[Dict[str, Any]], relevant_chunk_ids: Optional[List[str]], dialog: Any
    ) -> Dict[str, Any]:
        """
        Compute evaluation metrics for a single test case.

        Returns:
            Dictionary of metric names to values
        """
        metrics = {}

        # Retrieval metrics (if ground truth chunks provided)
        if relevant_chunk_ids:
            retrieved_ids = [c.get("chunk_id") for c in retrieved_chunks if c.get("chunk_id") is not None]
            metrics.update(cls.compute_retrieval_metrics(retrieved_ids, relevant_chunk_ids))

        # Generation metrics
        if generated_answer:
            # Basic metrics
            metrics["answer_length"] = len(generated_answer)
            metrics["has_answer"] = 1.0 if generated_answer.strip() else 0.0

            # Advanced metrics using LLM-as-judge
            llm_metrics = cls.evaluate_with_llm(question=question, answer=generated_answer, reference=reference_answer, retrieved_chunks=retrieved_chunks, dialog=dialog)
            metrics.update(llm_metrics)

        return metrics

    @classmethod
    def evaluate_with_llm(cls, question: str, answer: str, reference: Optional[str], retrieved_chunks: List[Dict[str, Any]], dialog: Any) -> Dict[str, Any]:
        """
        Evaluate answer quality using LLM-as-judge.

        Computes:
        - Faithfulness
        - Context Relevance
        - Answer Relevance
        - Semantic Similarity (if reference provided)
        """
        try:
            # Prepare context from retrieved chunks
            context_texts = []
            for chunk in retrieved_chunks:
                # Try to get content from various common keys
                text = chunk.get("content_with_weight") or chunk.get("content") or ""
                if text:
                    context_texts.append(text)
            context = "\n\n".join(context_texts)

            # Get LLM configuration from dialog
            tenant_id = dialog.tenant_id
            llm_id = dialog.llm_id

            # Use the same LLM as the dialog or a default chat model
            if TenantLLMService.llm_id2llm_type(llm_id) == LLMType.IMAGE2TEXT:
                chat_mdl = LLMBundle(tenant_id, LLMType.IMAGE2TEXT, llm_id)
            else:
                chat_mdl = LLMBundle(tenant_id, LLMType.CHAT, llm_id)

            # Prepare prompt
            prompt_template = load_prompt("evaluation_judge")
            prompt = PROMPT_JINJA_ENV.from_string(prompt_template).render(question=question, context=context, reference=reference if reference else "None", answer=answer)

            # Call LLM
            messages = [{"role": "user", "content": prompt}]

            # Fit message in context window
            _, msgs = message_fit_in(messages, chat_mdl.max_length)

            if not msgs:
                logging.warning("No messages generated for evaluation prompt.")
                return {"faithfulness": 0.0, "context_relevance": 0.0, "answer_relevance": 0.0, "semantic_similarity": None, "evaluation_status": "failed", "error": "Empty message list generated"}

            # Execute chat
            response = chat_mdl.chat(msgs[0]["content"], msgs[1:], {"temperature": 0.0})

            # Clean and parse response
            response = re.sub(r"^.*</think>", "", response, flags=re.DOTALL)
            response = re.sub(r"```json\s*", "", response)
            response = re.sub(r"```", "", response)

            metrics_json = json_repair.loads(response)

            # Validate and extract metrics
            valid_metrics = {}
            # Required float metrics
            for key in ["faithfulness", "context_relevance", "answer_relevance"]:
                val = metrics_json.get(key)
                if isinstance(val, (int, float)):
                    valid_metrics[key] = float(val)
                else:
                    valid_metrics[key] = 0.0

            # Optional/Nullable metrics
            val = metrics_json.get("semantic_similarity")
            if val is None:
                valid_metrics["semantic_similarity"] = None
            elif isinstance(val, (int, float)):
                valid_metrics["semantic_similarity"] = float(val)
            else:
                # If invalid type but not None, default to None seems safer given we want to avoid misleading 0.0
                valid_metrics["semantic_similarity"] = None

            # Optional explanations
            for key in ["faithfulness", "context_relevance", "answer_relevance", "semantic_similarity"]:
                expl_key = f"{key}_explanation"
                if expl_key in metrics_json:
                    valid_metrics[expl_key] = metrics_json[expl_key]

            # Status
            if "evaluation_status" in metrics_json:
                valid_metrics["evaluation_status"] = metrics_json["evaluation_status"]
            if "error" in metrics_json:
                valid_metrics["error"] = metrics_json["error"]

            return valid_metrics

        except Exception as e:
            logging.exception(f"Error in LLM evaluation: {e}")
            return {"faithfulness": 0.0, "context_relevance": 0.0, "answer_relevance": 0.0, "semantic_similarity": None, "evaluation_status": "failed", "error": str(e)}

    @classmethod
    def compute_retrieval_metrics(cls, retrieved_ids: List[str], relevant_ids: List[str]) -> Dict[str, float]:
        """
        Compute retrieval metrics.

        Args:
            retrieved_ids: List of retrieved chunk IDs
            relevant_ids: List of relevant chunk IDs (ground truth)

        Returns:
            Dictionary of retrieval metrics
        """
        if not relevant_ids:
            return {}

        retrieved_set = set(retrieved_ids)
        relevant_set = set(relevant_ids)

        # Precision: proportion of retrieved that are relevant
        precision = len(retrieved_set & relevant_set) / len(retrieved_set) if retrieved_set else 0.0

        # Recall: proportion of relevant that were retrieved
        recall = len(retrieved_set & relevant_set) / len(relevant_set) if relevant_set else 0.0

        # F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Hit rate: whether any relevant chunk was retrieved
        hit_rate = 1.0 if (retrieved_set & relevant_set) else 0.0

        # MRR (Mean Reciprocal Rank): position of first relevant chunk
        mrr = 0.0
        for i, chunk_id in enumerate(retrieved_ids, 1):
            if chunk_id in relevant_set:
                mrr = 1.0 / i
                break

        return {"precision": precision, "recall": recall, "f1_score": f1, "hit_rate": hit_rate, "mrr": mrr}

    @classmethod
    def compute_summary_metrics(cls, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute summary metrics across all test cases.

        Args:
            results: List of result dictionaries

        Returns:
            Summary metrics dictionary
        """
        if not results:
            return {}

        # Aggregate metrics
        metric_sums = {}
        metric_counts = {}

        for result in results:
            metrics = result.get("metrics", {})
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    metric_sums[key] = metric_sums.get(key, 0) + value
                    metric_counts[key] = metric_counts.get(key, 0) + 1

        # Compute averages
        summary = {"total_cases": len(results), "avg_execution_time": sum(r.get("execution_time", 0) for r in results) / len(results)}

        for key in metric_sums:
            summary[f"avg_{key}"] = metric_sums[key] / metric_counts[key]

        return summary
