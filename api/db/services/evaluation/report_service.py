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


import csv
import io
import json
import logging
from typing import List, Dict, Any, Optional, Tuple

from api.db.db_models import EvaluationRun, EvaluationResult, EvaluationCase


def sanitize_csv_cell(value: str) -> str:
    """
    Sanitize CSV cell to prevent formula injection.
    Prefixes values starting with =, +, -, @, or tab with a single quote.
    """
    if not value:
        return ""

    value = str(value)
    if value.startswith(("=", "+", "-", "@", "\t")):
        return f"'{value}"
    return value


class EvaluationReportService:
    @classmethod
    def get_run_results(cls, run_id: str) -> Tuple[bool, Dict[str, Any] | str]:
        """Get results for an evaluation run"""
        try:
            run = EvaluationRun.get_by_id(run_id)
            if not run:
                return False, "Evaluation run not found"

            results = EvaluationResult.select().where(EvaluationResult.run_id == run_id).order_by(EvaluationResult.create_time)

            return True, {"run": run.to_dict(), "results": [r.to_dict() for r in results]}
        except Exception as e:
            logging.error(f"Error getting run results {run_id}: {e}")
            return False, str(e)

    @classmethod
    def get_run_results_csv(cls, run_id: str) -> Optional[str]:
        """
        Get evaluation results as a CSV string.
        """
        try:
            # Verify run exists
            run = EvaluationRun.get_by_id(run_id)
            if not run:
                return None

            # Query results joined with case info
            query = (
                EvaluationResult.select(EvaluationResult, EvaluationCase)
                .join(EvaluationCase, on=(EvaluationResult.case_id == EvaluationCase.id))
                .where(EvaluationResult.run_id == run_id)
                .order_by(EvaluationResult.create_time)
            )

            # Materialize results to avoid re-executing query
            results_cache = list(query)

            # First pass: identify all metric keys
            all_metric_keys = set()
            for result in results_cache:
                metrics = result.to_dict().get("metrics", {})
                if metrics:
                    for k in metrics.keys():
                        all_metric_keys.add(f"metric_{k}")

            # Define CSV fields
            fieldnames = ["Question", "Reference Answer", "Generated Answer", "Execution Time"]
            fieldnames.extend(sorted(list(all_metric_keys)))
            fieldnames.extend(["Retrieved Chunks", "Relevant Chunk IDs"])

            # Generate CSV with streaming
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()

            # Second pass: use cached results
            for result in results_cache:
                result_dict = result.to_dict()
                case_dict = result.case_id.to_dict()

                # Sanitize user-controlled fields
                row = {
                    "Question": sanitize_csv_cell(case_dict.get("question", "")),
                    "Reference Answer": sanitize_csv_cell(case_dict.get("reference_answer", "")),
                    "Generated Answer": sanitize_csv_cell(result_dict.get("generated_answer", "")),
                    "Execution Time": result_dict.get("execution_time", 0),
                    "Retrieved Chunks": json.dumps(result_dict.get("retrieved_chunks", []), ensure_ascii=False),
                    "Relevant Chunk IDs": json.dumps(case_dict.get("relevant_chunk_ids", []), ensure_ascii=False),
                }

                # Handle metrics
                metrics = result_dict.get("metrics", {})
                if metrics:
                    for k, v in metrics.items():
                        row[f"metric_{k}"] = v

                writer.writerow(row)

            return output.getvalue()

        except Exception as e:
            logging.error(f"Error generating CSV for run {run_id}: {e}")
            return None

    @classmethod
    def get_recommendations(cls, run_id: str) -> Tuple[bool, List[Dict[str, Any]] | str]:
        """
        Analyze evaluation results and provide configuration recommendations.
        """
        try:
            run = EvaluationRun.get_by_id(run_id)
            if not run or not run.metrics_summary:
                return []

            metrics = run.metrics_summary
            recommendations = []

            # Low precision: retrieving irrelevant chunks
            if metrics.get("avg_precision", 1.0) < 0.7:
                recommendations.append(
                    {
                        "issue": "Low Precision",
                        "severity": "high",
                        "description": "System is retrieving many irrelevant chunks",
                        "suggestions": ["Increase similarity_threshold to filter out less relevant chunks", "Enable reranking to improve chunk ordering", "Reduce top_k to return fewer chunks"],
                    }
                )

            # Low recall: missing relevant chunks
            if metrics.get("avg_recall", 1.0) < 0.7:
                recommendations.append(
                    {
                        "issue": "Low Recall",
                        "severity": "high",
                        "description": "System is missing relevant chunks",
                        "suggestions": [
                            "Increase top_k to retrieve more chunks",
                            "Lower similarity_threshold to be more inclusive",
                            "Enable hybrid search (keyword + semantic)",
                            "Check chunk size - may be too large or too small",
                        ],
                    }
                )

            # Slow response time
            if metrics.get("avg_execution_time", 0) > 5.0:
                recommendations.append(
                    {
                        "issue": "Slow Response Time",
                        "severity": "medium",
                        "description": f"Average response time is {metrics['avg_execution_time']:.2f}s",
                        "suggestions": ["Reduce top_k to retrieve fewer chunks", "Optimize embedding model selection", "Consider caching frequently asked questions"],
                    }
                )

            return True, recommendations
        except Exception as e:
            logging.error(f"Error generating recommendations for run {run_id}: {e}")
            return False, str(e)

    @classmethod
    def compare_runs(cls, run_ids: List[str]) -> Tuple[bool, Dict[str, Any] | str]:
        """
        Compare multiple evaluation runs.
        """
        try:
            # Fetch runs
            runs_query = list(EvaluationRun.select().where(EvaluationRun.id.in_(run_ids)))
            runs_map = {r.id: r for r in runs_query}

            # Reorder according to input run_ids
            runs = []
            missing_ids = []
            for rid in run_ids:
                if rid in runs_map:
                    runs.append(runs_map[rid])
                elif rid not in missing_ids:  # Avoid duplicates in missing list
                    missing_ids.append(rid)

            if missing_ids:
                return False, f"Runs not found: {', '.join(missing_ids)}"

            # Check if all runs belong to the same dataset
            dataset_ids = {r.dataset_id_id for r in runs}
            if len(dataset_ids) > 1:
                return False, "Cannot compare runs from different datasets"

            # Prepare result structure
            run_details = [r.to_dict() for r in runs]
            comparison = {}

            # Pivot metrics
            all_metric_keys = set()
            for run in runs:
                if run.metrics_summary:
                    all_metric_keys.update(run.metrics_summary.keys())

            for key in all_metric_keys:
                comparison[key] = {}
                for run in runs:
                    if run.metrics_summary and key in run.metrics_summary:
                        comparison[key][run.id] = run.metrics_summary[key]

            return True, {"runs": run_details, "comparison": comparison}

        except Exception as e:
            logging.error(f"Error comparing runs: {e}")
            return False, str(e)
