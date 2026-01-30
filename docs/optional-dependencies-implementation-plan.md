# RAGflow Optional Dependencies & Plugin Architecture Implementation Plan

## Overview

This document outlines a phased approach to transition RAGflow from a monolithic dependency structure to a flexible plugin-based architecture with optional dependencies. This will enable:

- **Faster installations** for development and testing
- **Smaller Docker images** for production deployments
- **User choice** to install only needed features
- **Plugin extensibility** for community contributions

---

## Current State Analysis

### Dependency Count
- **Total dependencies:** 120+ packages in `pyproject.toml`
- **Installation time:** 10-15 minutes
- **Docker image size:** ~5-8 GB

### RAGflow Architecture (Important!)

RAGflow uses a **multi-database architecture**:

1. **PostgreSQL** (or MySQL) = **Application database** (REQUIRED)
   - User accounts, tenants, permissions
   - Knowledge base configurations
   - Document metadata (filename, size, processing status)
   - Conversation history, LLM settings
   - **NOT the vector embeddings or text chunks**

2. **Elasticsearch or OpenSearch** = **Vector database** (ONE REQUIRED)
   - Document chunks (actual text content)
   - Vector embeddings for semantic search
   - Full-text search index
   - **This is where RAG retrieval happens**

3. **MinIO** = **Object storage** (REQUIRED)
   - Original uploaded files (PDFs, DOCX, images, etc.)
   - **NOT cloud storage** - it's local S3-compatible storage
   - Acts as the file dump for all application files

4. **Redis** = **Cache/Sessions** (REQUIRED)
   - Session data, temporary caching

### Key Insight: Application DB Could Be Simplified

**Current:** PostgreSQL is required for application metadata  
**Opportunity:** Application metadata could use embedded SQLite for simpler deployments  
**Recommendation:** Make application database pluggable (SQLite/PostgreSQL/MySQL)

This would allow:
- **Simple deployments:** SQLite for personal/small team use
- **Production deployments:** PostgreSQL for multi-tenant/high-concurrency
- **Flexibility:** Users choose based on their needs

### Existing Plugin Patterns
✅ **Agent tools** already use graceful ImportError handling (`agent/tools/__init__.py`)  
✅ **Data connectors** use lazy imports (Jira, Slack)  
⚠️ **Document parsers** may have hard dependencies (Aspose)  
⚠️ **Frontend** needs audit for hardcoded feature references

---

## Three-Phase Implementation Plan

### Phase 1: Optional Dependencies (1-2 days)
**Goal:** Reorganize `pyproject.toml` without code changes

**Risk:** Very low (backward compatible)  
**Effort:** Minimal  
**Value:** Immediate install time savings

#### Tasks

1. **Categorize dependencies** into logical groups:
   - Core (always required)
   - Vector stores (PostgreSQL, Elasticsearch, OpenSearch, etc.)
   - LLM providers (OpenAI, Anthropic, Cohere, Google, etc.)
   - Cloud storage (AWS S3, Azure, Google Cloud, MinIO, etc.)
   - Document formats (PDF, Office, Email, etc.)
   - Data sources (Jira, Slack, GitHub, Confluence, etc.)
   - Search/Crawl (DuckDuckGo, Tavily, web scraping, etc.)
   - ML/Embeddings (ONNX, infinity-emb, etc.)

2. **Update `pyproject.toml`:**

```toml
[project]
dependencies = [
    # Core dependencies (always required)
    "flask>=3.0.0",
    "quart==0.20.0",
    "peewee==3.17.1",  # ORM (works with SQLite, PostgreSQL, MySQL)
    "minio==7.2.4",  # Object storage (REQUIRED - not optional)
    "valkey==6.0.2",  # Redis client (REQUIRED for caching)
    # ... other core deps
]

[project.optional-dependencies]
# Application Database (choose ONE - defaults to SQLite if none specified)
# SQLite is built into Python, no extra deps needed
db-postgres = ["psycopg2-binary>=2.9.11"]
db-mysql = ["pymysql>=1.1.1"]

# Vector Stores (choose ONE - required for RAG functionality)
vectorstore-elasticsearch = ["elasticsearch-dsl==8.12.0"]
vectorstore-opensearch = ["opensearch-py==2.7.1"]
vectorstore-postgres = ["psycopg2-binary>=2.9.11", "pyobvector==0.2.22"]  # pgvector extension
vectorstore-all = ["ragflow[vectorstore-elasticsearch]", "ragflow[vectorstore-opensearch]", "ragflow[vectorstore-postgres]"]

# LLM providers
llm-openai = ["openai>=1.45.0"]
llm-anthropic = ["anthropic==0.34.1"]
llm-cohere = ["cohere==5.6.2"]
llm-google = ["google-genai>=1.41.0", "vertexai==1.70.0"]
llm-all = ["ragflow[llm-openai]", "ragflow[llm-anthropic]", "ragflow[llm-cohere]", "ragflow[llm-google]"]

# Cloud storage (OPTIONAL - alternatives to MinIO for object storage)
storage-s3 = ["mypy-boto3-s3==1.40.26"]
storage-azure = ["azure-identity==1.17.1", "azure-storage-file-datalake==12.16.0"]
storage-gcs = ["google-api-python-client>=2.150.0", "google-auth-oauthlib>=1.2.0"]
storage-all = ["ragflow[storage-s3]", "ragflow[storage-azure]", "ragflow[storage-gcs]"]

# Document formats
docs-pdf = ["pdfplumber==0.10.4", "pypdf>=6.6.2", "pypdf2>=3.0.1"]
docs-office = ["python-docx>=1.1.2", "python-pptx>=1.0.2", "aspose-slides==24.7.0"]
docs-email = ["extract-msg>=0.39.0"]
docs-all = ["ragflow[docs-pdf]", "ragflow[docs-office]", "ragflow[docs-email]"]

# Understanding RAGflow's Database Architecture
# ============================================
# RAGflow uses TWO separate databases for different purposes:
#
# 1. APPLICATION DATABASE (choose one):
#    - SQLite (default, no extra install) - Best for: personal use, development, small teams
#    - PostgreSQL (db-postgres) - Best for: production, multi-tenant, high concurrency
#    - MySQL (db-mysql) - Best for: existing MySQL infrastructure
#    Stores: users, configs, document metadata, conversation history
#
# 2. VECTOR DATABASE (choose one, REQUIRED):
#    - Elasticsearch (vectorstore-elasticsearch) - Best for: full-text + vector search
#    - OpenSearch (vectorstore-opensearch) - Best for: open-source alternative to ES
#    - PostgreSQL+pgvector (vectorstore-postgres) - Best for: single-DB simplicity
#    Stores: document chunks, embeddings, semantic search index
#
# Installation Examples:
# ----------------------
# Simplest (SQLite + Elasticsearch):
#   pip install -e ".[vectorstore-elasticsearch]"
#
# Production (PostgreSQL for both):
#   pip install -e ".[db-postgres,vectorstore-postgres]"
#
# Hybrid (SQLite app + PostgreSQL vectors):
#   pip install -e ".[vectorstore-postgres]"
#
# Your current setup (PostgreSQL app + Elasticsearch vectors):
#   pip install -e ".[db-postgres,vectorstore-elasticsearch]"

# Data sources / Integrations
integrations-jira = ["jira==3.10.5"]
integrations-slack = ["slack-sdk==3.37.0"]
integrations-github = ["pygithub>=2.8.1"]
integrations-confluence = ["atlassian-python-api==4.0.7"]
integrations-all = ["ragflow[integrations-jira]", "ragflow[integrations-slack]", "ragflow[integrations-github]", "ragflow[integrations-confluence]"]

# Search and crawl
search-web = ["duckduckgo-search>=7.2.0", "tavily-python==0.5.1"]
search-crawl = ["Crawl4AI>=0.4.0", "selenium-wire==5.1.0", "webdriver-manager==4.0.1"]
search-all = ["ragflow[search-web]", "ragflow[search-crawl]"]
# ML and embeddings
ml-embeddings = ["infinity-sdk==0.7.0-dev2", "infinity-emb>=0.0.66"]
ml-onnx = [
    "onnxruntime==1.23.2; sys_platform == 'darwin' or platform_machine == 'x86_64'",
    "onnxruntime-gpu==1.23.2; sys_platform != 'darwin' and platform_machine in ('x86_64', 'AMD64')",
]
ml-all = ["ragflow[ml-embeddings]", "ragflow[ml-onnx]"]  # Note: This definition corrects a previously missing extras group

# Convenience groups
all = [
    "ragflow[vectorstore-all]",
    "ragflow[llm-all]",
    "ragflow[storage-all]",
    "ragflow[docs-all]",
    "ragflow[integrations-all]",
    "ragflow[search-all]",
    "ragflow[ml-all]",
]

[dependency-groups]
test = [
    "hypothesis>=6.132.0",
    "pytest>=8.3.5",
    "pytest-asyncio>=1.3.0",
    "pytest-xdist>=3.8.0",
    "pytest-cov>=7.0.0",
]
```

3. **Update Dockerfile** to support build-time selection:

```dockerfile
# builder stage
FROM base AS builder
USER root

WORKDIR /ragflow

# Add build argument for optional features
ARG RAGFLOW_EXTRAS="all"

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,id=ragflow_uv,target=/root/.cache/uv,sharing=locked \
    if [ "$NEED_MIRROR" == "1" ]; then \
        sed -i 's|pypi.org|pypi.tuna.tsinghua.edu.cn|g' uv.lock; \
    else \
        sed -i 's|pypi.tuna.tsinghua.edu.cn|pypi.org|g' uv.lock; \
    fi; \
    # Install based on RAGFLOW_EXTRAS
    if [ -z "$RAGFLOW_EXTRAS" ]; then \
        uv sync --python 3.12 --frozen --no-group test; \
    else \
        uv sync --python 3.12 --frozen --no-group test --extra "$RAGFLOW_EXTRAS"; \
    fi; \
    uv pip install pip==24.3.1
```

4. **Add SQLite Support** (NEW FEATURE - ~30 minutes):

> **Note:** SQLite is NOT currently supported in RAGflow. This would be a new feature, but thanks to excellent abstraction in the database layer, it's very easy to add!

**Why SQLite works with minimal changes:**
- Peewee ORM already supports SQLite
- Your models (`api/db/models/*.py`) are database-agnostic
- `init_database_tables()` automatically generates SQL for any database
- Transaction layer already handles SQLite (line 48 in `transaction.py`)

**Required changes:**

```python
# api/db/pool.py - Add SQLite to enum (5 lines)
from peewee import SqliteDatabase

class PooledDatabase(Enum):
    MYSQL = RetryingPooledMySQLDatabase
    POSTGRES = RetryingPooledPostgresqlDatabase
    SQLITE = SqliteDatabase  # No pooling needed - SQLite is file-based
```

```python
# api/db/connection.py - Update config logic (15 lines)
def get_database_config():
    db_type = db_type_raw.lower()
    
    if db_type == "sqlite":
        # SQLite only needs a file path
        return {
            "type": "sqlite",
            "name": database_config.get("name", "ragflow.db")
        }
    # ... existing postgres/mysql logic

def ensure_database_exists():
    config = get_database_config()
    
    if config["type"] == "sqlite":
        return  # SQLite creates DB file automatically
    # ... existing postgres/mysql logic

class BaseDataBase:
    def __init__(self):
        if settings.DATABASE_TYPE.upper() == "SQLITE":
            # SQLite doesn't need connection pooling
            from peewee import SqliteDatabase
            db_name = settings.DATABASE.get("name", "ragflow.db")
            self.database_connection = SqliteDatabase(db_name)
        else:
            # Existing pooled database logic
            ensure_database_exists()
            # ... existing code
```

```yaml
# conf/service_conf.yaml.template - Add SQLite option
# Option 1: SQLite (simplest - no server needed)
sqlite:
  name: 'ragflow.db'  # Database file path

# Option 2: PostgreSQL (production)
postgres:
  name: 'ragflow'
  user: 'ragflow'
  password: ${POSTGRES_PASSWORD}
  host: 'postgres-dev'
  port: 5432
  # ... existing config
```

**Total effort:** ~20 lines of code, 30 minutes

5. **Update documentation** (README.md):

```markdown
## Installation

### Full Installation (all features)
```bash
pip install -e ".[all]"
```

### Minimal Installation (core only, SQLite)
```bash
pip install -e .
# Uses SQLite by default - no database server needed!
```

### Custom Installation (specific features)
```bash
# PostgreSQL app DB + Elasticsearch vectors + OpenAI
pip install -e ".[db-postgres,vectorstore-elasticsearch,llm-openai]"

# All vector stores + all LLMs (SQLite for app data)
pip install -e ".[vectorstore-all,llm-all]"
```

### Docker Builds
```bash
# Full image (default)
docker build -t ragflow:full .

# Minimal image (SQLite + Elasticsearch)
docker build --build-arg RAGFLOW_EXTRAS="vectorstore-elasticsearch" -t ragflow:minimal .

# Custom image
docker build --build-arg RAGFLOW_EXTRAS="db-postgres,vectorstore-postgres,llm-openai" -t ragflow:custom .
```
```

5. **Test backward compatibility:**
   - `pip install -e ".[all]"` should work exactly as before
   - All existing Docker builds should continue working
   - CI/CD pipelines should not break

#### Deliverables
- ✅ Updated `pyproject.toml` with optional dependency groups
- ✅ Updated `Dockerfile` with `ARG RAGFLOW_EXTRAS`
- ✅ Updated README.md with installation instructions
- ✅ Backward compatibility verified

---

### Phase 2: Feature Detection & Graceful Degradation (1-2 weeks)
**Goal:** Make the application aware of available features and handle missing dependencies gracefully

**Risk:** Medium (requires code changes)  
**Effort:** Moderate  
**Value:** Better UX, no crashes from missing deps

#### Tasks

1. **Create feature detection module:**

```python
# config/features.py
"""
Detect which optional features are actually available.
This runs once at startup.
"""

AVAILABLE_FEATURES = {
    # Vector stores
    'vectorstore_postgres': False,
    'vectorstore_elasticsearch': False,
    'vectorstore_opensearch': False,
    
    # LLM providers
    'llm_openai': False,
    'llm_anthropic': False,
    'llm_cohere': False,
    'llm_google': False,
    
    # Integrations
    'integration_jira': False,
    'integration_slack': False,
    'integration_github': False,
    
    # Search
    'search_web': False,
    'search_crawl': False,
    
    # Document formats
    'docs_office': False,
    'docs_pdf': False,
}

def detect_features():
    """Check which optional dependencies are installed"""
    
    # Vector stores
    try:
        import psycopg2
        AVAILABLE_FEATURES['vectorstore_postgres'] = True
    except ImportError:
        pass
    
    try:
        import elasticsearch_dsl
        AVAILABLE_FEATURES['vectorstore_elasticsearch'] = True
    except ImportError:
        pass
    
    # LLM providers
    try:
        import openai
        AVAILABLE_FEATURES['llm_openai'] = True
    except ImportError:
        pass
    
    try:
        import anthropic
        AVAILABLE_FEATURES['llm_anthropic'] = True
    except ImportError:
        pass
    
    # Integrations
    try:
        import jira
        AVAILABLE_FEATURES['integration_jira'] = True
    except ImportError:
        pass
    
    try:
        import slack_sdk
        AVAILABLE_FEATURES['integration_slack'] = True
    except ImportError:
        pass
    
    # Search
    try:
        import duckduckgo_search
        AVAILABLE_FEATURES['search_web'] = True
    except ImportError:
        pass
    
    # Document formats
    try:
        import aspose.slides
        AVAILABLE_FEATURES['docs_office'] = True
    except ImportError:
        pass
    
    return AVAILABLE_FEATURES

# Run at module import
detect_features()
```

2. **Create API endpoint for feature discovery:**

```python
# api/apps/features_app.py
from flask import Blueprint
from config.features import AVAILABLE_FEATURES

features_bp = Blueprint('features', __name__)

@features_bp.route('/api/features', methods=['GET'])
def get_available_features():
    """Return which optional features are available in this deployment"""
    return {
        'features': AVAILABLE_FEATURES,
        'version': '0.23.1'
    }
```

3. **Add defensive checks to API endpoints:**

```python
# api/apps/datasource_app.py
from config.features import AVAILABLE_FEATURES

@app.route('/api/datasource/jira/connect', methods=['POST'])
def connect_jira():
    if not AVAILABLE_FEATURES['integration_jira']:
        return {
            'error': 'Jira integration not available',
            'message': 'Install with: pip install ragflow[integrations-jira]',
            'feature': 'integration_jira',
            'available': False
        }, 501  # Not Implemented
    
    # Safe to import now
    from jira import JIRA
    # ... actual implementation
```

4. **Update frontend to query available features:**

```typescript
// web/src/hooks/useFeatures.ts
import { useEffect, useState } from 'react';

interface Features {
  vectorstore_postgres: boolean;
  vectorstore_elasticsearch: boolean;
  integration_jira: boolean;
  integration_slack: boolean;
  search_web: boolean;
  // ... etc
}

export const useFeatures = () => {
  const [features, setFeatures] = useState<Features | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/api/features')
      .then(res => res.json())
      .then(data => {
        setFeatures(data.features);
        setLoading(false);
      });
  }, []);

  return { features, loading };
};
```

5. **Update UI components to conditionally render:**

```tsx
// web/src/pages/user-setting/data-source/index.tsx
import { useFeatures } from '@/hooks/useFeatures';

const DataSourcePage = () => {
  const { features, loading } = useFeatures();

  if (loading) return <Spinner />;

  return (
    <div>
      <h1>Available Data Sources</h1>
      
      {features?.integration_jira && (
        <DataSourceCard name="Jira" icon={JiraIcon} />
      )}
      
      {features?.integration_slack && (
        <DataSourceCard name="Slack" icon={SlackIcon} />
      )}
      
      {features?.integration_github && (
        <DataSourceCard name="GitHub" icon={GitHubIcon} />
      )}
      
      {!features?.integration_jira && 
       !features?.integration_slack && 
       !features?.integration_github && (
        <EmptyState message="No integrations installed. Install with: pip install ragflow[integrations-all]" />
      )}
    </div>
  );
};
```

6. **Add helpful error messages:**

```python
# common/exceptions.py
class FeatureNotAvailableError(Exception):
    """Raised when an optional feature is not installed"""
    
    def __init__(self, feature_name: str, install_command: str):
        self.feature_name = feature_name
        self.install_command = install_command
        super().__init__(
            f"Feature '{feature_name}' is not available. "
            f"Install with: {install_command}"
        )
```

#### Deliverables
- ✅ Feature detection module (`config/features.py`)
- ✅ `/api/features` endpoint
- ✅ Defensive checks in all API endpoints using optional deps
- ✅ Frontend hook for feature detection
- ✅ UI components conditionally rendered
- ✅ Helpful error messages for missing features

---

### Phase 3: Full Plugin Architecture (1-3 months)
**Goal:** True plugin system with discovery, clean abstractions, and extensibility

**Risk:** High (major refactor)  
**Effort:** Significant  
**Value:** Dream architecture - easy to extend, community contributions

#### Architecture Design

##### Plugin Registry Pattern

```python
# core/plugin_registry.py
from typing import Dict, Optional, Protocol, Type

class VectorStorePlugin(Protocol):
    """Interface for vector store plugins"""
    name: str
    
    def query(self, vector: list[float], filters: dict, top_k: int) -> list[dict]:
        ...
    
    def insert(self, documents: list[dict]) -> None:
        ...

class LLMPlugin(Protocol):
    """Interface for LLM plugins"""
    name: str
    
    def chat(self, messages: list[dict], model: str, **kwargs) -> str:
        ...
    
    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        ...

class PluginRegistry:
    _vectorstores: Dict[str, VectorStorePlugin] = {}
    _llms: Dict[str, LLMPlugin] = {}
    _integrations: Dict[str, any] = {}
    
    @classmethod
    def register_vectorstore(cls, name: str, plugin: VectorStorePlugin):
        cls._vectorstores[name] = plugin
    
    @classmethod
    def register_llm(cls, name: str, plugin: LLMPlugin):
        cls._llms[name] = plugin
    
    @classmethod
    def get_vectorstore(cls, name: str) -> Optional[VectorStorePlugin]:
        return cls._vectorstores.get(name)
    
    @classmethod
    def get_llm(cls, name: str) -> Optional[LLMPlugin]:
        return cls._llms.get(name)
    
    @classmethod
    def list_vectorstores(cls) -> list[str]:
        return list(cls._vectorstores.keys())
    
    @classmethod
    def list_llms(cls) -> list[str]:
        return list(cls._llms.keys())
```

##### Plugin Structure

```
plugins/
├── __init__.py
├── vectorstore/
│   ├── __init__.py
│   ├── postgres.py
│   ├── elasticsearch.py
│   └── opensearch.py
├── llm/
│   ├── __init__.py
│   ├── openai.py
│   ├── anthropic.py
│   ├── cohere.py
│   └── google.py
├── integrations/
│   ├── __init__.py
│   ├── jira.py
│   ├── slack.py
│   └── github.py
└── search/
    ├── __init__.py
    ├── duckduckgo.py
    └── tavily.py
```

##### Example Plugin Implementation

```python
# plugins/vectorstore/postgres.py
try:
    import psycopg2
    from pyobvector import ObVector
    from core.plugin_registry import PluginRegistry
    
    class PostgresVectorStore:
        name = "postgres"
        
        def __init__(self, connection_string: str):
            self.conn = psycopg2.connect(connection_string)
        
        def query(self, vector: list[float], filters: dict, top_k: int) -> list[dict]:
            # Implementation
            pass
        
        def insert(self, documents: list[dict]) -> None:
            # Implementation
            pass
    
    # Auto-register when imported
    PluginRegistry.register_vectorstore('postgres', PostgresVectorStore)
    
except ImportError:
    # Plugin not available - gracefully skip
    pass
```

##### Plugin Discovery

```python
# core/plugin_loader.py
import importlib
import pkgutil
from pathlib import Path

def discover_plugins():
    """Auto-discover and load all available plugins"""
    plugins_dir = Path(__file__).parent.parent / 'plugins'
    
    for finder, name, ispkg in pkgutil.iter_modules([str(plugins_dir)]):
        if ispkg:
            try:
                # Import the package, which triggers plugin registration
                importlib.import_module(f'plugins.{name}')
                print(f"✓ Loaded plugin category: {name}")
            except ImportError as e:
                print(f"✗ Plugin category '{name}' not available: {e}")

# Call at application startup
def init_plugins():
    """Initialize plugin system"""
    discover_plugins()
    
    from core.plugin_registry import PluginRegistry
    print(f"Available vector stores: {PluginRegistry.list_vectorstores()}")
    print(f"Available LLMs: {PluginRegistry.list_llms()}")
```

#### Tasks

1. **Define plugin interfaces** (Protocols) for each category
2. **Refactor existing implementations** to match plugin interfaces
3. **Create plugin registry** with discovery mechanism
4. **Move integrations** into plugin structure
5. **Update API layer** to use plugin registry
6. **Add plugin configuration** system
7. **Create plugin documentation** for contributors
8. **Add plugin testing framework**

#### Deliverables
- ✅ Plugin registry and discovery system
- ✅ Plugin interfaces (Protocols) for all categories
- ✅ All existing features refactored as plugins
- ✅ Plugin configuration system
- ✅ Plugin developer documentation
- ✅ Plugin testing framework

---

## Installation Examples

### Development

```bash
# Minimal for development
pip install -e ".[vectorstore-postgres,llm-openai]"

# With testing tools
uv sync --group test --extra "vectorstore-postgres,llm-openai"
```

### Production Docker

```bash
# Full-featured production image
docker build -t ragflow:prod .

# Minimal production image (only what you use)
docker build \
  --build-arg RAGFLOW_EXTRAS="vectorstore-postgres,llm-openai,docs-pdf" \
  -t ragflow:prod-minimal .
```

### Docker Compose

```yaml
# docker-compose.yml
services:
  ragflow:
    build:
      context: .
      args:
        RAGFLOW_EXTRAS: ${RAGFLOW_EXTRAS:-all}
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DATABASE_URL=${DATABASE_URL}
```

```bash
# .env
RAGFLOW_EXTRAS=vectorstore-postgres,llm-openai,docs-pdf
OPENAI_API_KEY=sk-proj-...
DATABASE_URL=postgresql://...
```

---

## Testing Strategy

### Phase 1 Testing
- ✅ Install with `[all]` - should work exactly as before
- ✅ Install with minimal deps - verify what breaks
- ✅ Install with custom combinations - test specific use cases
- ✅ Docker builds with different `RAGFLOW_EXTRAS` values

### Phase 2 Testing
- ✅ Feature detection returns correct values
- ✅ API endpoints return 501 for unavailable features
- ✅ Frontend hides unavailable features
- ✅ Error messages are helpful and actionable
- ✅ No crashes from missing dependencies

### Phase 3 Testing
- ✅ Plugin discovery finds all available plugins
- ✅ Plugin registry correctly manages plugins
- ✅ Plugins can be added/removed without core changes
- ✅ Plugin interfaces are well-defined and stable
- ✅ Community can contribute new plugins

---

## Migration Path for Users

### Current Users (No Changes Required)
```bash
# Existing installations continue working
pip install -e .
docker build -t ragflow .
```

### Users Wanting Faster Installs
```bash
# Install only what you need
pip install -e ".[vectorstore-postgres,llm-openai]"
```

### Users Deploying to Production
```bash
# Build smaller Docker images
docker build --build-arg RAGFLOW_EXTRAS="vectorstore-postgres,llm-openai" -t ragflow:prod .
```

---

## Success Metrics

### Phase 1
- ✅ Installation time reduced by 50-70% for minimal installs
- ✅ Docker image size reduced by 40-60% for custom builds
- ✅ Zero breaking changes for existing users

### Phase 2
- ✅ No crashes from missing dependencies
- ✅ Clear error messages for unavailable features
- ✅ Frontend adapts to available features

### Phase 3
- ✅ Clean plugin architecture
- ✅ Easy to add new providers
- ✅ Community can contribute plugins
- ✅ Core codebase is smaller and more maintainable

---

## Timeline Estimate

| Phase | Duration | Effort | Risk |
|-------|----------|--------|------|
| Phase 1 | 1-2 days | Low | Very Low |
| Phase 2 | 1-2 weeks | Medium | Medium |
| Phase 3 | 1-3 months | High | High |

**Recommended Approach:** Start with Phase 1, evaluate results, then decide on Phase 2/3.

---

## Next Steps

1. **Review this plan** and adjust based on priorities
2. **Start Phase 1** - reorganize `pyproject.toml`
3. **Test with minimal installs** - discover what actually breaks
4. **Document findings** - update this plan based on learnings
5. **Decide on Phase 2/3** - based on Phase 1 results

---

## Questions to Consider

- Which features do YOU actually use? (Helps prioritize groupings)
- What's your primary deployment method? (Docker, pip, uv)
- Do you want to support community plugins? (Affects Phase 3 design)
- What's your tolerance for breaking changes? (Affects timeline)
- Do you need backward compatibility? (Affects implementation)

---

## SQLite Support - Deferred

**Original estimate:** ~30 minutes, 20 lines of code
**Actual finding:** 20-30 days for production-ready implementation

### Blockers Identified

1. **Advisory locking (`api/db/locks.py`)** uses MySQL/PostgreSQL-specific functions
   - `GET_LOCK()` / `RELEASE_LOCK()` for MySQL
   - `pg_advisory_lock()` / `pg_advisory_unlock()` for PostgreSQL
   - SQLite has no equivalent - requires alternative locking strategy

2. **Connection pooling** incompatible with SQLite file-based model
   - Current architecture assumes pooled database connections
   - SQLite uses file-based locking, not connection pooling

3. **60+ migrations** need validation for SQLite DDL differences
   - SQLite has limited ALTER TABLE support
   - Many migration patterns may not translate directly

4. **Error handling** uses database-specific SQLSTATE codes
   - Exception handling tuned for MySQL/PostgreSQL error codes
   - Would need SQLite-specific error mapping

### Recommendation

Treat SQLite as a separate initiative, not part of Phase 1. The current PostgreSQL/MySQL architecture is well-suited for production deployments, and the optional dependencies feature provides significant value without SQLite support.
