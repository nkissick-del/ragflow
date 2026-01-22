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
import threading
from enum import Enum

from pydantic import BaseModel
from typing import Any, Optional
from common.config_utils import read_config
from urllib.parse import urlparse


class BaseConfig(BaseModel):
    id: int
    name: str
    host: str
    port: int
    service_type: str
    detail_func_name: str

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "host": self.host, "port": self.port, "service_type": self.service_type}


class ServiceConfigs:
    configs = list[BaseConfig]

    def __init__(self):
        self.configs = []
        self.lock = threading.Lock()


SERVICE_CONFIGS = ServiceConfigs


class ServiceType(Enum):
    METADATA = "metadata"
    RETRIEVAL = "retrieval"
    MESSAGE_QUEUE = "message_queue"
    RAGFLOW_SERVER = "ragflow_server"
    TASK_EXECUTOR = "task_executor"
    FILE_STORE = "file_store"


class MetaConfig(BaseConfig):
    meta_type: str

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        if "extra" not in result:
            result["extra"] = dict()
        extra_dict = result["extra"].copy()
        extra_dict["meta_type"] = self.meta_type
        result["extra"] = extra_dict
        return result


class MySQLConfig(MetaConfig):
    username: str
    password: str

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        if "extra" not in result:
            result["extra"] = dict()
        extra_dict = result["extra"].copy()
        extra_dict["username"] = self.username
        extra_dict["password"] = self.password
        result["extra"] = extra_dict
        return result


class PostgresConfig(MetaConfig):
    username: Optional[str] = None
    password: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        if "extra" not in result:
            result["extra"] = dict()
        extra_dict = result["extra"].copy()
        extra_dict["username"] = self.username
        extra_dict["password"] = self.password
        result["extra"] = extra_dict
        return result


class RetrievalConfig(BaseConfig):
    retrieval_type: str

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        if "extra" not in result:
            result["extra"] = dict()
        extra_dict = result["extra"].copy()
        extra_dict["retrieval_type"] = self.retrieval_type
        result["extra"] = extra_dict
        return result


class InfinityConfig(RetrievalConfig):
    db_name: str

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        if "extra" not in result:
            result["extra"] = dict()
        extra_dict = result["extra"].copy()
        extra_dict["db_name"] = self.db_name
        result["extra"] = extra_dict
        return result


class ElasticsearchConfig(RetrievalConfig):
    username: str
    password: str

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        if "extra" not in result:
            result["extra"] = dict()
        extra_dict = result["extra"].copy()
        extra_dict["username"] = self.username
        extra_dict["password"] = self.password
        result["extra"] = extra_dict
        return result


class MessageQueueConfig(BaseConfig):
    mq_type: str

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        if "extra" not in result:
            result["extra"] = dict()
        extra_dict = result["extra"].copy()
        extra_dict["mq_type"] = self.mq_type
        result["extra"] = extra_dict
        return result


class RedisConfig(MessageQueueConfig):
    database: int
    password: str

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        if "extra" not in result:
            result["extra"] = dict()
        extra_dict = result["extra"].copy()
        extra_dict["database"] = self.database
        extra_dict["password"] = self.password
        result["extra"] = extra_dict
        return result


class RabbitMQConfig(MessageQueueConfig):
    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        if "extra" not in result:
            result["extra"] = dict()
        return result


class RAGFlowServerConfig(BaseConfig):
    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        if "extra" not in result:
            result["extra"] = dict()
        return result


class TaskExecutorConfig(BaseConfig):
    message_queue_type: str

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        if "extra" not in result:
            result["extra"] = dict()
        result["extra"]["message_queue_type"] = self.message_queue_type
        return result


class FileStoreConfig(BaseConfig):
    store_type: str

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        if "extra" not in result:
            result["extra"] = dict()
        extra_dict = result["extra"].copy()
        extra_dict["store_type"] = self.store_type
        result["extra"] = extra_dict
        return result


class MinioConfig(FileStoreConfig):
    user: str
    password: str

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        if "extra" not in result:
            result["extra"] = dict()
        extra_dict = result["extra"].copy()
        extra_dict["user"] = self.user
        extra_dict["password"] = self.password
        result["extra"] = extra_dict
        return result


class DoclingConfig(BaseConfig):
    base_url: str

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        if "extra" not in result:
            result["extra"] = dict()
        result["extra"]["base_url"] = self.base_url
        return result


class OceanBaseConfig(MetaConfig):
    username: str
    password: str

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        if "extra" not in result:
            result["extra"] = dict()
        extra_dict = result["extra"].copy()
        extra_dict["username"] = self.username
        extra_dict["password"] = self.password
        result["extra"] = extra_dict
        return result


class UserDefaultLLMConfig(BaseConfig):
    default_models: dict

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        if "extra" not in result:
            result["extra"] = dict()
        result["extra"]["default_models"] = self.default_models
        return result


def load_configurations(config_path: str) -> list[BaseConfig]:
    raw_configs = read_config(config_path)
    configurations = []
    ragflow_count = 0
    id_count = 0
    for k, v in raw_configs.items():
        match k:
            case "ragflow":
                name: str = f"ragflow_{ragflow_count}"
                host: str = v["host"]
                http_port: int = v["http_port"]
                config = RAGFlowServerConfig(id=id_count, name=name, host=host, port=http_port, service_type="ragflow_server", detail_func_name="check_ragflow_server_alive")
                configurations.append(config)
                id_count += 1
            case "es":
                name: str = "elasticsearch"
                url = v.get("hosts", "")
                parsed = urlparse(url)
                host = parsed.hostname
                # Default to port 9200 if not specified (same as OpenSearch)
                try:
                    port: int = parsed.port or int(v.get("port") or 9200)
                except (ValueError, TypeError):
                    port = 9200
                if not host or not isinstance(port, int):
                    logging.warning(f"Invalid ES url: {url}, expected scheme://host:port")
                    continue
                username: str = v.get("username")
                password: str = v.get("password")
                config = ElasticsearchConfig(
                    id=id_count,
                    name=name,
                    host=host,
                    port=port,
                    service_type="retrieval",
                    retrieval_type="elasticsearch",
                    username=username,
                    password=password,
                    detail_func_name="get_es_cluster_stats",
                )
                configurations.append(config)
                id_count += 1

            case "infinity":
                name: str = "infinity"
                uri = v.get("uri", "")
                host = uri
                port = 7997
                if ":" in uri:
                    parts = uri.split(":", 1)
                    host = parts[0]
                    try:
                        port = int(parts[1])
                    except ValueError:
                        logging.warning(f"Invalid Infinity port in uri: {uri}")
                        continue

                # Validate host is non-empty before creating config
                if not host or not host.strip():
                    logging.warning(f"Skipping Infinity config: empty host (id={id_count}, name={name}, uri={uri})")
                    continue

                database: str = v.get("db_name", "default_db")
                config = InfinityConfig(id=id_count, name=name, host=host, port=port, service_type="retrieval", retrieval_type="infinity", db_name=database, detail_func_name="get_infinity_status")
                configurations.append(config)
                id_count += 1
            case "minio":
                name: str = "minio"
                url = v["host"]
                parts = url.split(":", 1)
                host = parts[0]
                port = int(parts[1])
                user = v.get("user")
                password = v.get("password")
                config = MinioConfig(id=id_count, name=name, host=host, port=port, user=user, password=password, service_type="file_store", store_type="minio", detail_func_name="check_minio_alive")
                configurations.append(config)
                id_count += 1
            case "redis":
                name: str = "redis"
                url = v["host"]
                parts = url.split(":", 1)
                host = parts[0]
                port = int(parts[1])
                password = v.get("password")
                db: int = v.get("db")
                config = RedisConfig(id=id_count, name=name, host=host, port=port, password=password, database=db, service_type="message_queue", mq_type="redis", detail_func_name="get_redis_info")
                configurations.append(config)
                id_count += 1
            case "mysql":
                name: str = "mysql"
                host: str = v.get("host")
                port: int = v.get("port")
                username = v.get("user")
                password = v.get("password")
                config = MySQLConfig(
                    id=id_count, name=name, host=host, port=port, username=username, password=password, service_type="meta_data", meta_type="mysql", detail_func_name="get_mysql_status"
                )
                configurations.append(config)
                id_count += 1
            case "admin":
                pass
            case "docling":
                name: str = "docling"
                try:
                    url = v.get("base_url", "")
                    parsed = urlparse(url)
                    host = parsed.hostname or url
                    port = parsed.port or 80
                    config = DoclingConfig(id=id_count, name=name, host=host, port=port, base_url=url, service_type="docling", detail_func_name="check_docling_alive")  # Placeholder function name
                    configurations.append(config)
                    id_count += 1
                except Exception as e:
                    logging.warning(f"Error parsing docling config: {e}")
            case "os":
                name: str = "opensearch"
                url = v.get("hosts", "")
                parsed = urlparse(url)
                host: str = parsed.hostname or v.get("host") or ""
                try:
                    port: int = parsed.port or int(v.get("port") or 9200)
                except (ValueError, TypeError):
                    port = 9200
                username: str = v.get("username")
                password: str = v.get("password")
                config = ElasticsearchConfig(
                    id=id_count, name=name, host=host, port=port, service_type="retrieval", retrieval_type="opensearch", username=username, password=password, detail_func_name="get_es_cluster_stats"
                )  # Reuse existing check
                configurations.append(config)
                id_count += 1
            case "postgres":
                name: str = "postgres"
                host: str = v.get("host")
                port: int = v.get("port")
                username = v.get("user")
                password = v.get("password")
                config = PostgresConfig(
                    id=id_count, name=name, host=host, port=port, username=username, password=password, service_type="meta_data", meta_type="postgres", detail_func_name="get_postgres_status"
                )  # Placeholder
                configurations.append(config)
                id_count += 1
            case "oceanbase":
                name: str = "oceanbase"
                conf = v.get("config", {})
                host: str = conf.get("host")
                port: int = int(conf.get("port", 0))
                username = conf.get("user")
                password = conf.get("password")
                config = OceanBaseConfig(
                    id=id_count, name=name, host=host, port=port, username=username, password=password, service_type="meta_data", meta_type="oceanbase", detail_func_name="get_oceanbase_status"
                )  # Placeholder
                configurations.append(config)
                id_count += 1
            case "user_default_llm":
                name: str = "user_default_llm"
                config = UserDefaultLLMConfig(id=id_count, name=name, host="", port=0, default_models=v.get("default_models", {}), service_type="llm_config", detail_func_name="")
                configurations.append(config)
                id_count += 1
            case "task_executor":
                name: str = "task_executor"
                host: str = v.get("host", "")
                port: int = v.get("port", 0)
                message_queue_type: str = v.get("message_queue_type")
                config = TaskExecutorConfig(
                    id=id_count, name=name, host=host, port=port, message_queue_type=message_queue_type, service_type="task_executor", detail_func_name="check_task_executor_alive"
                )
                configurations.append(config)
                id_count += 1
            case _:
                logging.warning(f"Unknown configuration key: {k}")
                continue

    return configurations
