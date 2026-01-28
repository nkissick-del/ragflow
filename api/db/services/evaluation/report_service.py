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


class EvaluationReportService:
    @classmethod
    def get_run_results(cls, run_id: str) -> Dict[str, Any]:
        """Get results for an evaluation run"""
        try:
            run = EvaluationRun.get_by_id(run_id)
            if not run:
                return {}

            results = EvaluationResult.select().where(EvaluationResult.run_id == run_id).order_by(EvaluationResult.create_time)

            return {"run": run.to_dict(), "results": [r.to_dict() for r in results]}
        except Exception as e:
            logging.error(f"Error getting run results {run_id}: {e}")
            return {}

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

            # Fetch data to memory to determine all metric keys
            rows = []
            all_metric_keys = set()

            for result in query:
                result_dict = result.to_dict()
                case_dict = result.case_id.to_dict()

                # Combine info
                row = {
                    "Question": case_dict.get("question", ""),
                    "Reference Answer": case_dict.get("reference_answer", ""),
                    "Generated Answer": result_dict.get("generated_answer", ""),
                    "Execution Time": result_dict.get("execution_time", 0),
                    "Retrieved Chunks": json.dumps(result_dict.get("retrieved_chunks", []), ensure_ascii=False),
                    "Relevant Chunk IDs": json.dumps(case_dict.get("relevant_chunk_ids", []), ensure_ascii=False),
                }

                # Handle metrics
                metrics = result_dict.get("metrics", {})
                if metrics:
                    for k, v in metrics.items():
                        metric_key = f"metric_{k}"
                        row[metric_key] = v
                        all_metric_keys.add(metric_key)

                rows.append(row)

            # Define CSV fields
            fieldnames = ["Question", "Reference Answer", "Generated Answer", "Execution Time"]

            # Add sorted metric keys
            fieldnames.extend(sorted(list(all_metric_keys)))

            # Add complex fields at the end
            fieldnames.extend(["Retrieved Chunks", "Relevant Chunk IDs"])

            # Generate CSV
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=fieldnames)

            writer.writeheader()
            for row in rows:
                writer.writerow(row)

            return output.getvalue()

        except Exception as e:
            logging.error(f"Error generating CSV for run {run_id}: {e}")
            return None

    @classmethod
    def get_recommendations(cls, run_id: str) -> List[Dict[str, Any]]:
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

            return recommendations
        except Exception as e:
            logging.error(f"Error generating recommendations for run {run_id}: {e}")
            return []

    @classmethod
    def compare_runs(cls, run_ids: List[str]) -> Tuple[bool, Dict[str, Any] | str]:
        """
        Compare multiple evaluation runs.
        """
        try:
            # Fetch runs
            runs = list(EvaluationRun.select().where(EvaluationRun.id.in_(run_ids)))

            # Check if all runs exist
            found_ids = {r.id for r in runs}
            missing_ids = set(run_ids) - found_ids
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
