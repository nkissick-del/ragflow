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

import time
import asyncio
import logging
import queue
import threading
from datetime import datetime
from timeit import default_timer as timer
from typing import Dict, Any, Optional, Tuple, List

from api.db.db_models import EvaluationRun, EvaluationResult, EvaluationDataset
from api.db.services.dialog_service import DialogService, async_chat
from api.db.services.evaluation.dataset_service import EvaluationDatasetService
from api.db.services.evaluation.metrics_service import EvaluationMetricsService
from common.misc_utils import get_uuid
from common.time_utils import current_timestamp


_SENTINEL = object()


def _sync_from_async_gen(async_gen, timeout=60, total_timeout=None):
    """
    Consumes an async generator in a background thread and yields items synchronously.

    Args:
        async_gen: The async generator to consume.
        timeout (int): Timeout in seconds for each read from the queue (per-item timeout).
        total_timeout (int, optional): Global timeout in seconds for the entire operation.
    """
    result_queue = queue.Queue(maxsize=10)
    stop_event = threading.Event()
    start_time = time.monotonic()

    def runner():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def consume():
            try:
                async for item in async_gen:
                    if stop_event.is_set():
                        break
                    while True:
                        try:
                            result_queue.put_nowait(item)
                            break
                        except queue.Full:
                            if stop_event.is_set():
                                return
                            await asyncio.sleep(0.1)
            except Exception as e:
                result_queue.put(e)
            finally:
                result_queue.put(_SENTINEL)

        try:
            loop.run_until_complete(consume())
        except Exception:
            pass
        finally:
            try:
                # Cancel all running tasks
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            finally:
                loop.close()

    threading.Thread(target=runner, daemon=True).start()

    try:
        while True:
            try:
                # Calculate timeout for this read
                wait_time = timeout
                if total_timeout is not None:
                    elapsed = time.monotonic() - start_time
                    remaining = total_timeout - elapsed
                    if remaining <= 0:
                        stop_event.set()
                        raise RuntimeError(f"Async generator total timeout after {total_timeout} seconds")
                    # Respect total_timeout
                    wait_time = min(remaining, timeout)

                item = result_queue.get(timeout=wait_time)
                if item is _SENTINEL:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
            except queue.Empty:
                stop_event.set()
                timeout_type = "total" if total_timeout is not None and (time.monotonic() - start_time) >= total_timeout else "per-read"
                raise RuntimeError(f"Async generator ({timeout_type}) timed out after {wait_time} seconds (timeout={timeout}, total_timeout={total_timeout})")
    except GeneratorExit:
        stop_event.set()
        raise


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
    def list_runs(cls, tenant_id: str, dataset_id: Optional[str] = None, dialog_id: Optional[str] = None, page: int = 1, page_size: int = 20) -> Tuple[List[Dict[str, Any]], int]:
        """
        List evaluation runs with tenant isolation and filtering.
        """
        try:
            query = EvaluationRun.select().join(EvaluationDataset, on=(EvaluationRun.dataset_id == EvaluationDataset.id)).where(EvaluationDataset.tenant_id == tenant_id)

            if dataset_id:
                query = query.where(EvaluationRun.dataset_id == dataset_id)
            if dialog_id:
                query = query.where(EvaluationRun.dialog_id == dialog_id)

            total = query.count()
            runs = list(query.order_by(EvaluationRun.create_time.desc()).paginate(page, page_size).dicts())

            return runs, total
        except Exception as e:
            logging.error(f"Error listing evaluation runs: {e}")
            return [], 0

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
            processed = 0
            total_cases = len(test_cases)

            for case in test_cases:
                result = cls.evaluate_single_case(run_id, case, dialog)
                if result:
                    results.append(result)

                processed += 1
                if processed % 10 == 0:
                    try:
                        EvaluationRun.update(metrics_summary={"progress": processed, "total": total_cases}).where(EvaluationRun.id == run_id).execute()
                        logging.info(f"Evaluation run {run_id} progress: {processed}/{total_cases}")
                    except Exception as e:
                        logging.warning(f"Failed to update progress for run {run_id}: {e}")
                        EvaluationRun.mark_failed(run_id)
                        return

            # Final 100% update
            try:
                EvaluationRun.update(metrics_summary={"progress": total_cases, "total": total_cases}).where(EvaluationRun.id == run_id).execute()
            except Exception as e:
                logging.warning(f"Failed to update completion progress for run {run_id}: {e}")
                EvaluationRun.mark_failed(run_id)
                return

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
            # Validate case ID
            case_id = case.get("id")
            if not case_id:
                logging.error(f"Test case missing ID: {case}")
                return None

            # Prepare messages
            messages = [{"role": "user", "content": case["question"]}]

            # Execute RAG pipeline
            start_time = timer()
            answer = ""
            retrieved_chunks = []
            token_usage = None

            def chat(dialog, messages, stream=True, **kwargs):
                return _sync_from_async_gen(async_chat(dialog, messages, stream=stream, **kwargs))

            # Consume single response for stream=False
            chat_gen = chat(dialog, messages, stream=False)
            try:
                ans = next(chat_gen)
                if isinstance(ans, dict):
                    answer = ans.get("answer", "")
                    retrieved_chunks = ans.get("reference", {}).get("chunks", [])

                    # Extract token usage
                    if "usage" in ans:
                        token_usage = ans["usage"]
                    elif "total_tokens" in ans:
                        token_usage = {"total_tokens": ans["total_tokens"]}
                else:
                    # Handle object response
                    # Update this branch to also populate answer and retrieved_chunks from the object by using getattr
                    answer = getattr(ans, "answer", None)
                    if answer is None:
                        answer = getattr(ans, "content", "")

                    if answer is None:
                        answer = ""

                    if not isinstance(answer, str):
                        answer = str(answer)

                    retrieved_chunks = getattr(ans, "retrieved_chunks", None)
                    if retrieved_chunks is None:
                        retrieved_chunks = getattr(ans, "chunks", [])
                    if not isinstance(retrieved_chunks, list):
                        retrieved_chunks = []

                    if getattr(ans, "usage", None):
                        token_usage = getattr(ans, "usage")
                    elif getattr(ans, "total_tokens", None):
                        token_usage = {"total_tokens": getattr(ans, "total_tokens")}

                # Final normalization: ensure token_usage is a dict if it has total_tokens attribute
                if token_usage and not isinstance(token_usage, dict) and hasattr(token_usage, "total_tokens"):
                    token_usage = {"total_tokens": token_usage.total_tokens}
            except StopIteration:
                logging.warning(f"Chat generator empty for case {case_id}")
            except Exception as e:
                logging.error(f"Error during chat generation for case {case_id}: {e}")

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
                "case_id": case_id,
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

            return result
        except Exception as e:
            logging.error(f"Error evaluating case {case.get('id')}: {e}")
            return None
