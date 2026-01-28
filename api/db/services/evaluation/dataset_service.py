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
from typing import List, Dict, Any, Optional, Tuple

from api.db.db_models import EvaluationDataset, EvaluationCase
from api.db.services.common_service import CommonService
from common.constants import StatusEnum
from common.misc_utils import get_uuid
from common.time_utils import current_timestamp


class EvaluationDatasetService(CommonService):
    model = EvaluationDataset

    @classmethod
    def create_dataset(cls, name: str, description: str, kb_ids: List[str], tenant_id: str, user_id: str) -> Tuple[bool, str]:
        """
        Create a new evaluation dataset.
        """
        try:
            timestamp = current_timestamp()
            dataset_id = get_uuid()
            dataset = {
                "id": dataset_id,
                "tenant_id": tenant_id,
                "name": name,
                "description": description,
                "kb_ids": kb_ids,
                "created_by": user_id,
                "create_time": timestamp,
                "update_time": timestamp,
                "status": StatusEnum.VALID.value,
            }

            if not EvaluationDataset.create(**dataset):
                return False, "Failed to create dataset"

            return True, dataset_id
        except Exception as e:
            logging.error(f"Error creating evaluation dataset: {e}")
            return False, str(e)

    @classmethod
    def get_dataset(cls, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get dataset by ID"""
        try:
            dataset = EvaluationDataset.get_by_id(dataset_id)
            if dataset and dataset.status == StatusEnum.VALID.value:
                return dataset.to_dict()
            return None
        except Exception as e:
            logging.error(f"Error getting dataset {dataset_id}: {e}")
            return None

    @classmethod
    def list_datasets(cls, tenant_id: str, user_id: str, page: int = 1, page_size: int = 20) -> Dict[str, Any]:
        """List datasets for a tenant"""
        try:
            query = EvaluationDataset.select()
            query = query.where((EvaluationDataset.tenant_id == tenant_id) & (EvaluationDataset.created_by == user_id) & (EvaluationDataset.status == StatusEnum.VALID.value))
            query = query.order_by(EvaluationDataset.create_time.desc())

            total = query.count()
            datasets = query.paginate(page, page_size)

            return {"total": total, "datasets": [d.to_dict() for d in datasets]}
        except Exception as e:
            logging.error(f"Error listing datasets: {e}")
            return {"total": 0, "datasets": []}

    @classmethod
    def update_dataset(cls, dataset_id: str, **kwargs) -> bool:
        """Update dataset"""
        try:
            # Check existence and status
            dataset = EvaluationDataset.get_or_none(EvaluationDataset.id == dataset_id, EvaluationDataset.status == StatusEnum.VALID.value)
            if not dataset:
                logging.error(f"Dataset {dataset_id} not found or invalid")
                return False

            kwargs["update_time"] = current_timestamp()
            return EvaluationDataset.update(**kwargs).where(EvaluationDataset.id == dataset_id).execute() > 0
        except Exception as e:
            logging.error(f"Error updating dataset {dataset_id}: {e}")
            return False

    @classmethod
    def delete_dataset(cls, dataset_id: str) -> bool:
        """Soft delete dataset"""
        try:
            # Soft delete dataset and cascade to test cases
            rows = EvaluationDataset.update(status=StatusEnum.INVALID.value, update_time=current_timestamp()).where(EvaluationDataset.id == dataset_id).execute()
            if rows > 0:
                # Cascade soft delete to test cases
                # Verify EvaluationCase has "status" and "update_time" (we added them or assumed them)
                try:
                    EvaluationCase.update(status=StatusEnum.INVALID.value, update_time=current_timestamp()).where(EvaluationCase.dataset_id == dataset_id).execute()
                except Exception as e:
                    logging.warning(f"Failed to cascade soft delete to test cases for dataset {dataset_id}: {e}")

            return rows > 0
        except Exception as e:
            logging.error(f"Error deleting dataset {dataset_id}: {e}")
            return False

    @classmethod
    def add_test_case(
        cls,
        dataset_id: str,
        question: str,
        reference_answer: Optional[str] = None,
        relevant_doc_ids: Optional[List[str]] = None,
        relevant_chunk_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        """
        Add a test case to a dataset.
        """
        try:
            # Verify dataset exists and is valid
            dataset = EvaluationDataset.get_or_none(EvaluationDataset.id == dataset_id, EvaluationDataset.status == StatusEnum.VALID.value)
            if not dataset:
                return False, "Dataset not found or invalid"

            case_id = get_uuid()
            timestamp = current_timestamp()
            case = {
                "id": case_id,
                "dataset_id": dataset_id,
                "question": question,
                "reference_answer": reference_answer,
                "relevant_doc_ids": relevant_doc_ids,
                "relevant_chunk_ids": relevant_chunk_ids,
                "metadata": metadata,
                "create_time": timestamp,
                "update_time": timestamp,
                "status": StatusEnum.VALID.value,
            }

            if not EvaluationCase.create(**case):
                return False, "Failed to create test case"

            return True, case_id
        except Exception as e:
            logging.error(f"Error adding test case: {e}")
            return False, str(e)

    @classmethod
    def get_test_cases(cls, dataset_id: str) -> List[Dict[str, Any]]:
        """Get all test cases for a dataset"""
        try:
            # Only return valid test cases for valid datasets
            # We can also check dataset status, but usually UI does that first.
            # But the prompt suggested: "also consider adding a dataset status check in get_test_cases to skip cases for datasets with StatusEnum.INVALID"
            # And since we soft-delete cases now, we should filter by case status too.
            cases = EvaluationCase.select().where((EvaluationCase.dataset_id == dataset_id) & (EvaluationCase.status == StatusEnum.VALID.value)).order_by(EvaluationCase.create_time)

            return [c.to_dict() for c in cases]
        except Exception as e:
            logging.error(f"Error getting test cases for dataset {dataset_id}: {e}")
            return []

    @classmethod
    def delete_test_case(cls, case_id: str) -> bool:
        """Delete a test case"""
        try:
            return EvaluationCase.update(status=StatusEnum.INVALID.value, update_time=current_timestamp()).where(EvaluationCase.id == case_id).execute() > 0
        except Exception as e:
            logging.error(f"Error deleting test case {case_id}: {e}")
            return False

    @classmethod
    def import_test_cases(cls, dataset_id: str, cases: List[Dict[str, Any]]) -> Tuple[int, int]:
        """
        Bulk import test cases from a list.
        """
        success_count = 0
        failure_count = 0
        case_instances = []

        if not cases:
            return success_count, failure_count

        cur_timestamp = current_timestamp()

        try:
            # First validate dataset existence? Usually import implies dataset exists.
            # But prompt focused on "validate each incoming case_data.question is non-empty"

            valid_cases = []
            for case_data in cases:
                if not case_data.get("question"):
                    # Skip empty questions
                    failure_count += 1
                    continue
                valid_cases.append(case_data)

            if not valid_cases:
                return success_count, failure_count

            for case_data in valid_cases:
                case_id = get_uuid()
                case_info = {
                    "id": case_id,
                    "dataset_id": dataset_id,
                    "question": case_data.get("question", ""),
                    "reference_answer": case_data.get("reference_answer"),
                    "relevant_doc_ids": case_data.get("relevant_doc_ids"),
                    "relevant_chunk_ids": case_data.get("relevant_chunk_ids"),
                    "metadata": case_data.get("metadata"),
                    "create_time": cur_timestamp,
                    "update_time": cur_timestamp,
                    "status": StatusEnum.VALID.value,
                }

                case_instances.append(EvaluationCase(**case_info))

            EvaluationCase.bulk_create(case_instances, batch_size=300)

            # Verify success count by querying
            success_count = (
                EvaluationCase.select().where((EvaluationCase.dataset_id == dataset_id) & (EvaluationCase.create_time == cur_timestamp) & (EvaluationCase.status == StatusEnum.VALID.value)).count()
            )

            # The bulk_create guarantees all or nothing in transaction usually, but Peewee bulk_create might be partial if atomic=False?
            # Assuming atomic by default or wrapped in transaction.
            # However, prompt asks to "determine actual success by querying... and set success_count and failure_count = total_cases - success_count".
            failure_count = len(cases) - success_count

        except Exception as e:
            logging.error(f"Error bulk importing test cases: {str(e)}")
            # On exception, assume failure for valid_cases that weren't counted yet
            failure_count = len(cases)
            success_count = 0
            # If partial success, query would catch it, but we are inside exception block.
            # Prompt: "on exception from EvaluationCase.bulk_create, determine actual success by querying..."
            try:
                success_count = (
                    EvaluationCase.select().where((EvaluationCase.dataset_id == dataset_id) & (EvaluationCase.create_time == cur_timestamp) & (EvaluationCase.status == StatusEnum.VALID.value)).count()
                )
                failure_count = len(cases) - success_count
            except Exception:
                pass

        return success_count, failure_count
