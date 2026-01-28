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

"""
RAG Evaluation Service

Provides functionality for evaluating RAG system performance including:
- Dataset management
- Test case management
- Evaluation execution
- Metrics computation
- Configuration recommendations
"""

from typing import List, Dict, Any, Optional, Tuple

from api.db.services.common_service import CommonService
from api.db.db_models import EvaluationDataset
from api.db.services.evaluation.dataset_service import EvaluationDatasetService
from api.db.services.evaluation.runner_service import EvaluationRunnerService
from api.db.services.evaluation.report_service import EvaluationReportService


class EvaluationService(CommonService):
    """Service for managing RAG evaluations (Facade)"""

    model = EvaluationDataset

    # ==================== Dataset Management ====================

    @classmethod
    def create_dataset(cls, name: str, description: str, kb_ids: List[str], tenant_id: str, user_id: str) -> Tuple[bool, str]:
        return EvaluationDatasetService.create_dataset(name, description, kb_ids, tenant_id, user_id)

    @classmethod
    def get_dataset(cls, dataset_id: str, tenant_id: str = None) -> Optional[Dict[str, Any]]:
        dataset = EvaluationDatasetService.get_dataset(dataset_id)
        if not dataset:
            return None
        if tenant_id and dataset.get("tenant_id") != tenant_id:
            return None
        return dataset

    @classmethod
    def list_datasets(cls, tenant_id: str, user_id: str, page: int = 1, page_size: int = 20) -> Dict[str, Any]:
        return EvaluationDatasetService.list_datasets(tenant_id, user_id, page, page_size)

    @classmethod
    def update_dataset(cls, dataset_id: str, **kwargs) -> bool:
        return EvaluationDatasetService.update_dataset(dataset_id, **kwargs)

    @classmethod
    def delete_dataset(cls, dataset_id: str) -> bool:
        return EvaluationDatasetService.delete_dataset(dataset_id)

    # ==================== Test Case Management ====================

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
        return EvaluationDatasetService.add_test_case(
            dataset_id,
            question,
            reference_answer,
            relevant_doc_ids,
            relevant_chunk_ids,
            metadata,
        )

    @classmethod
    def get_test_cases(cls, dataset_id: str, page: int = 1, page_size: int = 20) -> Tuple[List[Dict[str, Any]], int]:
        return EvaluationDatasetService.get_test_cases(dataset_id, page, page_size)

    @classmethod
    def delete_test_case(cls, case_id: str) -> bool:
        return EvaluationDatasetService.delete_test_case(case_id)

    @classmethod
    def import_test_cases(cls, dataset_id: str, cases: List[Dict[str, Any]]) -> Tuple[int, int]:
        return EvaluationDatasetService.import_test_cases(dataset_id, cases)

    # ==================== Evaluation Execution ====================

    @classmethod
    def start_evaluation(cls, dataset_id: str, dialog_id: str, user_id: str, name: Optional[str] = None) -> Tuple[bool, str]:
        return EvaluationRunnerService.start_evaluation(dataset_id, dialog_id, user_id, name)

    @classmethod
    def list_runs(cls, tenant_id: str, dataset_id: Optional[str] = None, dialog_id: Optional[str] = None, page: int = 1, page_size: int = 20) -> Tuple[List[Dict[str, Any]], int]:
        return EvaluationRunnerService.list_runs(tenant_id, dataset_id, dialog_id, page, page_size)

    # ==================== Results & Analysis ====================

    @classmethod
    def get_run_results(cls, run_id: str) -> Tuple[bool, Dict[str, Any] | str]:
        return EvaluationReportService.get_run_results(run_id)

    @classmethod
    def get_run_results_csv(cls, run_id: str) -> Optional[str]:
        return EvaluationReportService.get_run_results_csv(run_id)

    @classmethod
    def get_recommendations(cls, run_id: str) -> Tuple[bool, List[Dict[str, Any]] | str]:
        return EvaluationReportService.get_recommendations(run_id)

    @classmethod
    def compare_runs(cls, run_ids: List[str]) -> Tuple[bool, Dict[str, Any] | str]:
        return EvaluationReportService.compare_runs(run_ids)
