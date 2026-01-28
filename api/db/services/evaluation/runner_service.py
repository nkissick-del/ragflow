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

import asyncio
import logging
import queue
import threading
from datetime import datetime
from timeit import default_timer as timer
from typing import Dict, Any, Optional, Tuple

from api.db.db_models import EvaluationRun, EvaluationResult
from api.db.services.dialog_service import DialogService
from api.db.services.evaluation.dataset_service import EvaluationDatasetService
from api.db.services.evaluation.metrics_service import EvaluationMetricsService
from common.misc_utils import get_uuid
from common.time_utils import current_timestamp


def _sync_from_async_gen(async_gen, timeout=60):
    result_queue = queue.Queue()

    def runner():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def consume():
            try:
                async for item in async_gen:
                    result_queue.put(item)
            except Exception as e:
                result_queue.put(e)
            finally:
                result_queue.put(StopIteration)

        try:
            loop.run_until_complete(consume())
        except Exception:
            pass
        finally:
            loop.close()

    threading.Thread(target=runner, daemon=True).start()

    while True:
        try:
            item = result_queue.get(timeout=timeout)
            if item is StopIteration:
                break
            if isinstance(item, Exception):
                raise item
            yield item
        except queue.Empty:
            raise RuntimeError(f"Async generator timed out after {timeout} seconds")


class EvaluationRunnerService:
    @classmethod
    def start_evaluation(cls, dataset_id: str, dialog_id: str, user_id: str, name: Optional[str] = None) -> Tuple[bool, str]:
        """
        Start an evaluation run.
        """
        try:
            # Validate dataset
            if not EvaluationDatasetService.get_dataset(dataset_id):
                return False, "Dataset not found"

            # Get dialog configuration
            success, dialog = DialogService.get_by_id(dialog_id)
            if not success:
                return False, "Dialog not found"

            # Create evaluation run
            run_id = get_uuid()
            if not name:
                name = f"Evaluation Run {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

            run = {
                "id": run_id,
                "dataset_id": dataset_id,
                "dialog_id": dialog_id,
                "name": name,
                "config_snapshot": dialog.to_dict(),
                "metrics_summary": None,
                "status": "RUNNING",
                "created_by": user_id,
                "create_time": current_timestamp(),
                "complete_time": None,
            }

            if not EvaluationRun.create(**run):
                return False, "Failed to create evaluation run"

            # Execute evaluation asynchronously
            threading.Thread(target=cls.execute_evaluation, args=(run_id, dataset_id, dialog)).start()

            return True, run_id
        except Exception as e:
            logging.error(f"Error starting evaluation: {e}")
            return False, str(e)

    @classmethod
    def execute_evaluation(cls, run_id: str, dataset_id: str, dialog: Any):
        """
        Execute evaluation for all test cases.
        """
        try:
            # Get all test cases
            test_cases = EvaluationDatasetService.get_test_cases(dataset_id)

            if not test_cases:
                EvaluationRun.update(status="FAILED", complete_time=current_timestamp()).where(EvaluationRun.id == run_id).execute()
                return

            # Execute each test case
            results = []
            for case in test_cases:
                result = cls.evaluate_single_case(run_id, case, dialog)
                if result:
                    results.append(result)

            # Check if any results were obtained
            if not results:
                logging.warning(f"No results generated for run {run_id}")
                EvaluationRun.update(status="FAILED", complete_time=current_timestamp()).where(EvaluationRun.id == run_id).execute()
                return

            # Compute summary metrics
            metrics_summary = EvaluationMetricsService.compute_summary_metrics(results)

            # Update run status
            status = "COMPLETED" if len(results) == len(test_cases) else "PARTIAL"
            EvaluationRun.update(status=status, metrics_summary=metrics_summary, complete_time=current_timestamp()).where(EvaluationRun.id == run_id).execute()

        except Exception as e:
            logging.error(f"Error executing evaluation {run_id}: {e}")
            EvaluationRun.update(status="FAILED", complete_time=current_timestamp()).where(EvaluationRun.id == run_id).execute()

    @classmethod
    def evaluate_single_case(cls, run_id: str, case: Dict[str, Any], dialog: Any) -> Optional[Dict[str, Any]]:
        """
        Evaluate a single test case.
        """
        try:
            # Prepare messages
            messages = [{"role": "user", "content": case["question"]}]

            # Execute RAG pipeline
            start_time = timer()
            answer = ""
            retrieved_chunks = []
            token_usage = None

            def chat(dialog, messages, stream=True, **kwargs):
                from api.db.services.dialog_service import async_chat

                return _sync_from_async_gen(async_chat(dialog, messages, stream=stream, **kwargs))

            # Consume single response for stream=False
            chat_gen = chat(dialog, messages, stream=False)
            try:
                ans = next(chat_gen)
                if isinstance(ans, dict):
                    answer = ans.get("answer", "")
                    retrieved_chunks = ans.get("reference", {}).get("chunks", [])

                    # Extract token usage
                    # Assuming ans might have 'usage' or we can calculate from what we have if the model returns it
                    # The user mentioned: "e.g., response.get("usage") or response.usage"
                    if "usage" in ans:
                        token_usage = ans["usage"]
                    elif "total_tokens" in ans:
                        # Some APIs handle it differently, but user asked to populate token_usage
                        token_usage = {"total_tokens": ans["total_tokens"]}
            except StopIteration:
                logging.warning(f"Chat generator empty for case {case.get('id')}")
            except Exception as e:
                logging.error(f"Error during chat generation for case {case.get('id')}: {e}")

            execution_time = timer() - start_time

            # Compute metrics
            metrics = EvaluationMetricsService.compute_metrics(
                question=case["question"],
                generated_answer=answer,
                reference_answer=case.get("reference_answer"),
                retrieved_chunks=retrieved_chunks,
                relevant_chunk_ids=case.get("relevant_chunk_ids"),
                dialog=dialog,
            )

            # Save result
            result_id = get_uuid()
            result = {
                "id": result_id,
                "run_id": run_id,
                "case_id": case["id"],
                "generated_answer": answer,
                "retrieved_chunks": retrieved_chunks,
                "metrics": metrics,
                "execution_time": execution_time,
                "token_usage": token_usage,
                "create_time": current_timestamp(),
            }

            try:
                EvaluationResult.create(**result)
            except Exception as e:
                logging.error(f"Failed to persist evaluation result {result_id}: {e}")
                # We still return the result so the run can continue and stats can be computed
                pass

            return result
        except Exception as e:
            logging.error(f"Error evaluating case {case.get('id')}: {e}")
            return None
