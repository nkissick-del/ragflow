# Architecture Proposal: RAGFlow Modular "Brain Transplant"

**Date:** 2026-01-22  
**Status:** DRAFT  
**Authors:** User & Antigravity

---

## 1. Executive Summary

Existing RAGFlow logic (DeepDoc) is tightly coupled: it conflates "Seeing the Document" (OCR/Layout) with "Formatting the Document" (Chunking). This forces high-fidelity parsers like Docling to be "dumbed down" into flat lists of text, which are then poorly re-assembled by naive token-counting algorithms.

**The Decision:** Retain the RAGFlow "Chassis" (UI, Auth, Database) but Gut the "Engine". Replace the legacy DeepDoc pipeline with a modular, decoupled architecture inspired by LlamaIndex and Unstructured.io.

---

## 2. The Problem: "The Template Trap"

Currently, the `General` template acts as a "Blender":

1. **Parser (Docling):** Extracts rich structure (Headers, Tables, Paragraphs).
2. **Adapter:** Flattens this into a list of strings (`splitlines()`), destroying structure.
3. **Template (General):** Naively glues strings back together until `token_limit=128`.

**Result:**
- Loss of Semantic Integrity (Headers merged into unrelated paragraphs)
- Inability to treat document types differently (Legal vs. Technical)
- Hard dependency on DeepDoc's visual-based layout recognition

---

## 3. The Vision: The "Semantic Tree" Architecture

We are moving from a monolith to a strict layered "Assembly Line" architecture, with DeepDOC quarantined as a pure engine.

### 3.1 Target Directory Topology

```text
ragflow/
├── deepdoc/                   # [QUARANTINED ENGINE]
│   ├── vision/                # Core CV library (OCR, Layout)
│   └── parser/                # Internal format logic. NO external access.
│
├── rag/
│   ├── parsers/               # [LAYER 1: INPUT ACQUISITION]
│   │   ├── base.py            # Common logic (Retry policies, API handling)
│   │   ├── docling_client.py  # Connects to Docling API/Container
│   │   ├── mineru_client.py   # Connects to MinerU API/Container
│   │   └── deepdoc_client.py  # Wrapper calling local `deepdoc` (The only bridge)
│   │
│   ├── orchestration/         # [LAYER 2: NORMALIZATION]
│   │   ├── base.py            # StandardizedDocument & Contracts
│   │   ├── normalize.py       # Main orchestration script
│   │   └── adapters/          # Adapters: specific parser output -> Standardized IR
│   │
│   └── templates/             # [LAYER 3: BUSINESS LOGIC]
│       ├── base.py            # Common chunking utilities
│       ├── semantic.py        # Structure-aware chunking
│       ├── legal.py           # Domain logic (Contracts)
│       └── ...
```

### 3.2 Philosophy: "Quarantine & Delegate"

1.  **DeepDOC at Root:** `deepdoc/` lives at the project root. It is treated as an **internal 3rd-party library**.
    *   **Strict Barrier:** No code in `rag/` may import `deepdoc` directly, EXCEPT `rag/parsers/deepdoc_client.py`.
    *   **Optionality:** If `deepdoc` dependencies are missing, `deepdoc_client.py` gracefully degrades (disables the parser).

2.  **Parsers as Clients:** The `rag/parsers/` directory contains **Clients**, not heavy logic.
    *   They connect to an "Engine" (Service, API, or Isolated Library).
    *   They return raw data (JSON/Markdown) to be normalized later.

3.  **DRY Roots:**
    *   `rag/parsers/base.py`: Handles HTTP retries, timeouts, and error reporting.
    *   `rag/orchestration/base.py`: Defines the `StandardizedDocument` contract.
    *   `rag/templates/base.py`: Handles token counting and text overlap logic.

### 3.3 The Microservice Future

This structure prepares RAGFlow to shed weight. Currently, "Engines" like Docling play the role of external microservices. Eventually, DeepDOC can also be peeled off into a sidecar container, leaving the `ragflow-core` container lightweight and focused purely on orchestration and retrieval.

---

## 4. Layer Responsibilities

### Layer 1: Parsers
| Attribute | Value |
|-----------|-------|
| **Role** | "Solve the File Format" |
| **Input** | Raw PDF, DOCX, HTML, etc. |
| **Components** | `docling_client.py`, `mineru_client.py`, `deepdoc_client.py` |
| **Output** | Parser-specific structured output (may vary in format) |
| **Rule** | Never makes business decisions (chunk size, domain logic) |

### Layer 2: Orchestration (Normalization)
| Attribute | Value |
|-----------|-------|
| **Role** | "Translate and Standardize" |
| **Input** | Parser-specific output |
| **Components** | `orchestrator.py` with parser-specific adapters |
| **Output** | **Standardized IR** (consistent format regardless of parser) |
| **Rule** | Parser-specific logic lives here, not in templates |

### Layer 3: Templates
| Attribute | Value |
|-----------|-------|
| **Role** | "Solve the Business Problem" |
| **Input** | Standardized IR (parser-agnostic) |
| **Components** | `templates/semantic.py`, `templates/aemo_isp.py`, `templates/legal.py` |
| **Output** | Optimized chunks for vector storage/retrieval |
| **Rule** | Domain-specific logic; **never knows which parser was used** |

### 4.4 Error Handling & Resilience
| Layer | Failure Mode | Responsibility & Strategy |
|-------|--------------|---------------------------|
| **Parsers** | API timeout, 429, Malformed PDF | **Retry/Backoff:** Exponential backoff (max_attempts=3, total_deadline=30s, initial_delay=1s, multiplier=2x, max_delay=5s). <br> **Fallback:** Fallback to MinerU/DeepDOC after N retries or immediately on "Malformed IR" validation failure/5xx errors. <br> **Output:** Emit explicit error object (not crash). |
| **Orchestration** | Invalid IR, Empty Output | **Validation:** Validate parser output against schema. <br> **Sanitization:** Strip corrupted chars. <br> **Partial Success:** Log warnings but proceed if partial content exists. |
| **Templates** | Missing Headers, Text-only | **Defensive Logic:** Default to "General" chunking if structure missing. <br> **Recovery:** Graceful degradation (don't crash on flat text). |

**Observability:**
- **Centralized Logging:** All layers log to standard ELK/Loki stack with `correlation_id` (propagated from API request).
- **User Notification:** UI Toast + Dashboard Alert: "Parsing partially failed - some formatting may be lost" (emitted by Orchestration layer).
- **Retry Policy:** Automatic retry for transient errors (Network/503/Timeout); fail-fast for 400s (Client Error).

---

## 5. The Standardized IR (Contract)

The "handshake" between the orchestration layer and templates:

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class DocumentElement:
    """A semantic element extracted from the document."""
    type: str  # "heading", "paragraph", "table", "code_block", "list", "image"
    content: str
    level: Optional[int] = None  # For headings: 1-6
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StandardizedDocument:
    """The contract between parsers and templates."""
    
    # 1. PRIMARY STORAGE: The full structured content (backing field)
    # Excluded from init to allow property setter to handle invalidation
    _content: str = field(default="", metadata={'serialize': True}) 
    
    # 2. METADATA: About the document
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 3. CACHED ELEMENTS: Internal storage for lazy parsing
    # Marked as init=False/repr=False to prevent serialization and direct instantiation
    _elements: Optional[List[DocumentElement]] = field(default=None, init=False, repr=False, metadata={'serialize': False})

    @property
    def content(self) -> str:
        return self._content

    @content.setter
    def content(self, value: str):
        """Reset cached elements when content changes to ensure consistency."""
        self._content = value
        self._elements = None  # Invalidate cache

    @property
    def elements(self) -> List[DocumentElement]:
        """
        Lazy derivation strategy:
        - If elements have been supplied (e.g. by adapter), return them.
        - If not, parse `self._content` on demand and cache the result.
        """
        if self._elements is None:
            self._elements = self._parse_elements(self._content)
        return self._elements

    def populate_elements(self, elements: List[DocumentElement]) -> None:
        """
        Validated population method for adapters to forcefully set elements.
        Replaces direct access to _elements.
        """
        # (Optional) Schema validation here
        self._elements = elements

    @content.setter
    def content(self, value: str):
        """Reset cached elements when content changes to ensure consistency."""
        self._content = value
        self._elements = None  # Invalidate cache

    def _parse_elements(self, content: str) -> List[DocumentElement]:
        # Implementation of on-demand parsing logic
        # NOTE: See LlamaIndex MarkdownNodeParser (Section 10) for parsing algorithm.
        # This is non-trivial and must validate/normalize info before caching.
        ...
```

**Usage Note:**
- **Precedence:** Templates should always access `.elements` (the property), never `_elements`.
- **Adapters:** If an adapter has efficient access to structure, it *should* populate `_elements` to save costs. If not, it can just set `content` and let the template parse it lazily.
- **Synchronization:** `content` is the source of truth. If you modify `content`, you must clear `_elements`.
- **Memory:** `elements` are transient and cached; they are not serialized to the DB. only `content` is stored.

All parsers (via their adapters) must produce this format. All templates consume this format.

---

## 6. Existing RAGFlow Infrastructure (Reuse)

> [!IMPORTANT]
> RAGFlow already has significant infrastructure we can leverage. We are NOT reinventing the wheel.

### ✅ Features We Can Reuse

| Feature | RAGFlow Location | How We Use It |
|---------|------------------|---------------|
| **Chunk Overlap** | `overlapped_percent` in `parser_config`, `naive_merge()` in `rag/nlp/__init__.py` | Reuse parameter; adapt overlap logic for semantic template |
| **Configurable Chunk Size** | `chunk_token_num` in `parser_config` | Reuse directly |
| **Metadata Storage** | `metadata` JSON column in `rag/utils/ob_conn.py` | Store `header_path` in existing metadata field |
| **Metadata Filtering** | `get_metadata_filter_expression()` in `ob_conn.py` | Filter by `header_path` at query time |
| **Document Titles** | `title_tks`, `docnm_kwd` fields | Continue using for document-level metadata |
| **Multi-Template System** | `rag/app/templates/*.py` (12 templates exist) | Add `semantic.py` as new template |
| **Delimiter Config** | `delimiter` in `parser_config` | Reuse for fallback splitting |

### ❌ Features We Are Adding (New)

| Feature | What It Does | LlamaIndex Reference |
|---------|--------------|----------------------|
| **Header Hierarchy Metadata** | Each chunk knows its path (`/H1/H2/H3/`) | `MarkdownNodeParser.header_path` |
| **Stack-Based Header Tracking** | Maintain hierarchy while parsing | `MarkdownNodeParser.header_stack` |
| **Structure-Aware Chunking** | Never split mid-paragraph; respect headers | `HierarchicalChunker.chunk()` |
| **Code Block Protection** | Don't parse headers inside ``` fences | `MarkdownNodeParser.code_block` flag |

### Metadata Schema & Migration Strategy

> [!NOTE]
> We are standardizing `header_path` as an **ordered array** to support precise hierarchy matching.
> **Store:** `metadata` JSONB column in `document` table (managed by `rag/utils/ob_conn.py`).

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `header_path` | `List[str]` | Ordered array of header strings. Depth 0 is root. | `["Introduction", "Background"]` (H1->H2) |
| `header_path_schema_version` | `str` | Schema version indicator | `"v1"` |

**Conflict Handling & Migration:**
1.  **Read (Legacy Compat):** If `header_path` is a scalar string (e.g., `"/A/B"`), split on legacy delimiter (e.g., `/`): `"/A/B".strip("/").split("/")` -> `["A", "B"]`. 
    *   **Limitation:** Fails if headers contain `"/"`. 
    *   **Mitigation:** Mitigation: Define escape scheme (e.g. `\/`) or use alternate delimiter if possible. Read-time normalization MUST handle both escaped legacy scalars and new JSON arrays (check `header_path_schema_version`).
2.  **Write:** Always write `header_path` as a JSON array and set `"header_path_schema_version": "v1"`.
3.  **Migration Job:** Background task iterates `document` table, updates scalar `header_path` to array format, adds version.
    *   **Timeline:** Non-blocking. Can run alongside active traffic.
    *   **Performance:** Batch size 1000. **Validate with DB-specific benchmarks** (JSONB update costs, index write amplification). Performance varies by DB load and metadata size.
    *   **Mixed State:** Readers must support both array and scalar formats during migration (read-time normalization).

**Query Pattern (Database Appropriate):**
-   **Postgres (JSONB):** `metadata->'header_path' @> '["Introduction"]'` (Containment) or `metadata->'header_path'->>0 = 'Introduction'` (Root match).
-   **Elasticsearch:** `term` query on `header_path.keyword` (exact match) or `nested` query for complex path logic.

---

## 7. Implementation Roadmap

### Phase 1: The Contract
| Goal | Stop `docling_parser.py` from destroying data |
|------|-----------------------------------------------|
| Action | Modify `docling_parser.py` to return structured Markdown (not `splitlines()`) |
| Artifact | Updated `deepdoc/parser/docling_parser.py` |

### Phase 2: The Orchestration Layer
| Goal | Create normalization adapters in the orchestrator |
|------|--------------------------------------------------|
| Action | Add `DoclingAdapter` to `orchestrator.py` that converts Docling output → Standardized IR |
| Future | Add `MinerUAdapter`, `DeepDOCAdapter` as needed |
| Artifact | Updated `rag/app/orchestrator.py` |

### Phase 3: The Semantic Template
| Goal | Create a parser-agnostic template that understands structure |
|------|-------------------------------------------------------------|
| Action | Create `rag/app/templates/semantic.py` with stack-based header tracking |
| Behavior | Chunks respecting boundaries; never splits paragraphs; preserves hierarchy metadata |
| Artifact | New `rag/app/templates/semantic.py` |

### Phase 4: Wiring
| Goal | Route structured parser outputs to the new template |
|------|-----------------------------------------------------|
| Action | Update routing logic: if parser produces structured IR → use `semantic` template |
| Artifact | Updated `rag/app/orchestrator.py`, `rag/app/format_parsers.py` |

### Phase 5: Migration & Rollout (Deployment)
| Goal | Safe transition from Legacy to Semantic pipeline |
|------|--------------------------------------------------|
| **Backwards Compatibility** | **Separate Route:** Do NOT use dual-mode DoclingAdapter. Implement a distinct semantic route. Legacy route remains untouched until deprecation. |
| **Routing** | Update `rag/app/orchestrator.py` to route traffic to `DoclingAdapter`/`semantic` template based on config/flag. |
| **Reprocessing Strategy** | **Lazy Consideration:** No lazy migration on-read. **Explicit Reprocessing Rules:** <br> - Docs with >10 queries in 30d (Track via new `document_query_metrics` table/service) <br> - Uploads after [Date] <br> - Specific MIME types (e.g. Contracts) |
| **Rollout Plan** | 1. **Prerequisite:** Verify RAGFlow telemetry supports per-document metrics (or implement before Phase 4). <br> 2. **Feature Flag:** Enabled for 1% of uploads. <br> 3. **Observation:** 48h soak time. Monitor: Parser failure rate, Chunk validation failures, Query-time errors. <br> 4. **Expand:** 10% -> 48h -> 50% -> 48h -> 100%. |
| **Rollback** | **Typed Thresholds:** <br> - Parsing Failures > 1% <br> - Chunk Validation Failures > 1% <br> - Query-time Errors > 1% <br> - **Data Corruption > 0.1% (Critical)** |
| **Monitoring** | **Baselines:** Est. 1 week before rollout. <br> **Window:** 7-day rolling average. <br> **Metrics:** Latency (p50/p95/p99), Parser Latency, Memory RSS. |
| **Targets** | `rag/app/orchestrator.py` (emit query events), `rag/app/templates/semantic.py`, `rag/app/format_parsers.py`, `db_migration_scripts`, `rag/config` (Feature Flags). |


---

## 8. Long-Term: Domain Templates

Once the semantic template works, adding domain-specific templates is trivial:

1. Copy `templates/semantic.py` → `templates/aemo_isp.py`
2. Add domain logic (e.g., look for "Decision", "Clause" headers)
3. Add as dropdown option in RAGFlow UI

**User selects:**
- **Parser:** Docling / MinerU / DeepDOC (independent choice)
- **Template:** General / AEMO ISP / Legal (independent choice)

---

## 9. Key Decisions Log

### Decision 1: Adapt LlamaIndex Patterns, Don't Import Library
**Date:** 2026-01-22 | **Status:** ✅ APPROVED

We will **adapt** LlamaIndex's patterns (stack-based header tracking) without importing the library.

### Decision 2: Templates are Parser-Agnostic
**Date:** 2026-01-22 | **Status:** ✅ APPROVED

Templates should be **domain-specific**, not parser-specific. The orchestration layer handles parser normalization.

### Decision 3: Common Orchestration Layer for Parser Translation
**Date:** 2026-01-22 | **Status:** ✅ APPROVED

A common orchestration layer with parser-specific adapters translates each parser's output into the Standardized IR.

### Decision 4: Reuse RAGFlow Infrastructure Where Possible
**Date:** 2026-01-22 | **Status:** ✅ APPROVED

Leverage existing RAGFlow features (overlap, metadata, chunk size, templates) rather than reimplementing.

---

## 10. LlamaIndex Reference Code

> [!IMPORTANT]
> **Development Rule:** When implementing any chunking/parsing logic, cross-reference against the LlamaIndex source code below to ensure alignment with industry patterns.

### Primary References

| Component | GitHub URL | What To Check |
|-----------|------------|---------------|
| **MarkdownNodeParser** | [markdown.py](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/node_parser/file/markdown.py) | Stack-based header tracking, code block protection |
| **DoclingNodeParser** | [base.py](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/node_parser/llama-index-node-parser-docling/llama_index/node_parser/docling/base.py) | How LlamaIndex consumes Docling output |
| **HierarchicalChunker** | [hierarchical_chunker.py](https://github.com/DS4SD/docling-core/blob/main/docling_core/transforms/chunker/hierarchical_chunker.py) | Docling's native chunking algorithm |
| **DoclingReader** | [base.py](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-docling/llama_index/readers/docling/base.py) | Markdown vs JSON export options |

### Algorithm: Stack-Based Header Tracking

```python
# From LlamaIndex MarkdownNodeParser
# CROSS-REFERENCE THIS DURING IMPLEMENTATION

header_stack: List[tuple[int, str]] = []  # (level, text)
code_block = False

for line in lines:
    # Track code blocks to avoid parsing headers inside them
    if line.lstrip().startswith("```"):
        code_block = not code_block
        current_section += line + "\n"
        continue

    if not code_block:
        header_match = re.match(r"^(#+)\s(.*)", line)
        if header_match:
            # Emit previous section as a chunk
            if current_section.strip():
                emit_node(current_section, header_stack)

            level = len(header_match.group(1))
            text = header_match.group(2)

            # Pop headers of equal or higher level (going "up" the hierarchy)
            while header_stack and header_stack[-1][0] >= level:
                header_stack.pop()

            header_stack.append((level, text))
            current_section = f"{'#' * level} {text}\n"
            continue

    current_section += line + "\n"

# Emit final section
if current_section.strip():
    emit_node(current_section, header_stack)
```

### Key Implementation Details from LlamaIndex

1. **Header Path Separator:** Default `/` (e.g., `/Introduction/Background/`)
2. **Metadata Key:** `header_path`
3. **Include Header in Chunk:** Yes, the header line is included in the chunk text
4. **Empty Sections:** Sections with only a header (no content) are still emitted

---

## 11. Future Enhancements (Post-MVP)

| Feature | Priority | LlamaIndex Reference |
|---------|----------|----------------------|
| **Prev/Next Relationships** | Nice-to-have | `include_prev_next_rel` parameter |
| **Parent/Child Hierarchy** | Future | `HierarchicalNodeParser` + `AutoMergingRetriever` |
| **Semantic Chunking** | Future | `SemanticSplitterNodeParser` (requires embedding at ingestion) |
| **LLM Metadata Extraction** | Future | `TitleExtractor`, `QuestionsAnsweredExtractor` |

### Footnotes & Citations (Deferred)

**Goal:** Extract footnotes, citations, and page numbers as searchable metadata.

**What Docling Supports (as of Jan 2026):**
- ✅ **Page numbers** — Available via `ProvenanceItem.page_no` for each `DocItem`
- ✅ **Footnote detection** — Footnotes are a recognized `DocItemLabel` type
- ❌ **Footnote ↔ reference linking** — NOT automatic (GitHub feature request exists Oct 2025)

**Implementation Options (when ready):**

| Approach | Difficulty | Description |
|----------|------------|-------------|
| **Page number per chunk** | ✅ Easy | Populate `metadata.page_no` from `ProvenanceItem` |
| **Footnotes as separate chunks** | ✅ Easy | Detect `DocItemLabel.FOOTNOTE`, add `is_footnote: True` metadata |
| **Regex-based linking** | ⚠️ Medium | Match `[1]` or superscript markers to footnote text (brittle) |
| **LLM citation parsing** | ⚠️ Medium | Use LLM to extract structured citation (author, year, title) from footnote text |
| **Full footnote ↔ reference linking** | ❌ Hard | Wait for Docling feature or build custom visual-position-based linking |

**Why Deferred:** Core semantic chunking must work first. The architecture already supports adding these metadata fields later—just populate more fields in the Standardized IR.

### Automatic Template Selection (Deferred)

**Goal:** Use an LLM call at the orchestration layer to automatically classify documents and select the most appropriate template without user intervention.

**How It Would Work:**
```python
# In orchestrator.py (future)
if parser_config.get("auto_select_template"):
    doc_sample = standardized_ir.content[:2000]  # First ~500 tokens
    classification = llm.classify(
        prompt=f"Classify this document type: {doc_sample}",
        options=["Legal", "Technical", "Academic", "Regulatory", "General"]
    )
    template = TEMPLATE_MAP[classification]  # e.g., "Legal" → "templates/legal.py"
```

**Considerations:**

| Aspect | Assessment |
|--------|------------|
| **Technical Difficulty** | ⚠️ Medium — Straightforward LLM classification |
| **Prerequisites** | Templates must exist first before auto-selecting between them |
| **Latency/Cost** | Adds LLM call at ingestion time (~1-2s, ~$0.001/doc) |
| **Architecture Fit** | ✅ Perfect — Orchestration layer is where this belongs |
| **Fallback** | If confidence < threshold, fall back to "General" template |

**Why Deferred:** Need working templates before we can auto-select between them. Phase 1-4 must complete first.

### Selective VLM Processing for Images/Charts (Deferred)

**Goal:** Route images/charts to an expensive multimodal VLM for description, while text uses standard embedding. Critical for economics documents with graphs/charts.

**What Already Exists in RAGFlow:**

| Component | Location | Status |
|-----------|----------|--------|
| `VisionFigureParser` | `deepdoc/parser/figure_parser.py` | ✅ Detects figures, routes to VLM |
| `vision_llm_figure_describe_prompt` | `rag/prompts/generator.py` | ✅ Prompts for figure description |
| Context injection | `figure_parser.py:150` | ✅ Passes surrounding text to VLM |
| Multiple VLM backends | `rag/llm/cv_model.py` | ✅ GPT-4V, Gemini Vision, Claude |

**What Docling Adds:**

| Feature | Docling Support |
|---------|-----------------|
| **Figure/chart detection** | ✅ `DocItemLabel.PICTURE` / `FIGURE` |
| **Image export** | ✅ Can export figure images with provenance |
| **Chart understanding** | ⚠️ "Coming soon" — Bar/pie/line parsing in development |
| **SmolDocling VLM** | ✅ Built-in vision model for captioning |

**Ideal Flow (Conceptual):**
```python
# In orchestrator.py (future)
for element in standardized_ir.elements:
    if element.type in ["IMAGE", "FIGURE", "CHART"]:
        # Expensive VLM call - only for images
        description = vlm.describe(
            element.image_bytes,
            context=element.surrounding_text,
            prompt="Describe this economics chart, including trends, axes, and key data points."
        )
        element.metadata["vlm_description"] = description
        element.text_for_embedding = description  # Embed description, not pixels
    else:
        # Standard text embedding - cheaper
        element.text_for_embedding = element.text
```

**Cost Optimization:**

| Content Type | Processing | Approximate Cost |
|--------------|------------|-----------------|
| Text | Standard embedding | ~$0.0001/1K tokens |
| Images/Charts | VLM description → then embed | ~$0.01-0.03/image |

**Implementation Options:**

| Approach | Difficulty | Description |
|----------|------------|-------------|
| **Use existing RAGFlow vision** | ✅ Easy | Already exists, may need better wiring |
| **Docling image extraction + VLM** | ⚠️ Medium | Extract via Docling, route to separate VLM |
| **Domain-specific prompts** | ⚠️ Medium | Custom prompts for economics/charts |
| **Wait for Docling chart parsing** | ⏳ Future | Native chart understanding is "coming soon" |

**Why Deferred:** Core semantic chunking must work first. The VLM infrastructure exists in RAGFlow; the gap is Docling integration and cost-optimized routing.

---

## 12. UI Changes

**New Template Dropdown Options:**

When Phase 3 is complete, the following new template(s) should be exposed in the RAGFlow UI:

| Template | Description | When Available |
|----------|-------------|----------------|
| **Semantic** | Structure-aware chunking with header hierarchy | Phase 3 |
| **AEMO ISP** | Regulatory document processing (example domain template) | Phase 5+ |
| **Legal** | Contract/legal document processing (example) | Phase 5+ |

**User Selection Model:**
- **Parser:** Docling / MinerU / DeepDOC (independent choice)
- **Template:** General / Semantic / Domain-specific (independent choice)

---

## 13. Next Steps

### Implementation Phases (Status Check)

- [x] **Phase 1:** Modify `docling_parser.py` to return structured Markdown (Implemented)
- [x] **Phase 2:** Add `DoclingAdapter` to orchestration layer (Implemented)
- [x] **Phase 3:** Create `templates/semantic.py` with header tracking (Implemented)
- [x] **Phase 4:** Update routing logic for structured parsers (Implemented)
- [ ] **Phase 5:** Expose "Semantic" template in UI dropdown (Pending Frontend)

### The New Default: "Semantic" vs "Templates"

**Update 2026-01-24:**
The `Semantic` template is intended to become the **New General** template (the default for all structure-aware parsers).
Future "Templates" (e.g. Legal, AEMO) will effectively be **Post-Processing Modules** that run *after* or *on top of* the semantic structure, tailoring the chunking strategy to specific domain rules.


### Testing Requirements


> [!IMPORTANT]
> Each phase must include testing before proceeding to the next phase.

| Category | Testing Approach & Acceptance Criteria |
|----------|----------------------------------------|
| **Phase 1** | **Unit test:** Verify `docling_parser.py` structure. <br> - Assert: H1-H6 detection, header level metadata, code blocks (```) don't parse inner headers, HTML tables -> Markdown. <br> - **Fixture:** `test/fixtures/semantic_test_corpus/sample.pdf` (5+ pages, 3 levels, 1 table, 1 code block). <br> -> **Tasks:** Create `test/benchmark/`, `test/resilience/` and seed fixtures. |
| **Phase 2** | **Unit test:** Verify `DoclingAdapter` converts Docling output → Standardized IR. <br> **Success:** IR matches schema. |
| **Phase 3** | **Integration test:** End-to-end test with sample PDF. <br> **Success:** Chunks in DB have correct `header_path` metadata. <br> **Lazy Loading:** Verify `_elements` caching behavior. |
| **Phase 4** | **Integration test:** Verify routing logic. <br> **Success:** Semantic template selected automatically. <br> **Context:** Verify `correlation_id` propagation. |
| **E2E Validation** | **Manual test:** Upload structured PDF via UI. <br> **Success:** Database chunks preserve hierarchy. |
| **Performance** | **Benchmark:** Measure throughput/memory. <br> **Target:** 3 pages/sec (Unified), <500MB RAM/job. |
| **Regression** | **Comparator:** Legacy vs Semantic on `test/fixtures/regression_corpus/` (>=50 docs, >=500 pages, 20% distribution tech/legal/tables/plain). <br> **Success:** 100% pass on existing tests. |
| **Error Resilience** | **Fuzzing:** Test with malformed PDFs, incomplete parser output. <br> **Success:** Graceful failure with logs (no crashes). |
| **Load Testing** | **Stress Test:** Simulate 100 concurrent uploads for 5 mins (500 total uploads). <br> **Stretch Goal:** Sustain 500 concurrent uploads. <br> **Success:** Stability, no timeouts. |
| **Security** | **Adversarial Test:** Malicious PDFs (zip bombs, deeply nested structures, JS injection). <br> **Metadata Injection:** Test SQL/NoSQL injection via header_path. <br> **Success:** Parser fails safely, no code execution, appropriate logs. |
| **Backward Compat** | **Read Test:** Metadata migration (Legacy -> v1). <br> **Feature Flag:** Verify toggle behavior. |

**Test Locations:**
- **Functional:** `test/integration/test_docling_integration.py`
- **Non-Functional:** `test/benchmark/`, `test/resilience/`

---

## 14. Implementation Audit (2026-01-24)

**Auditor:** Antigravity

### Status Overview
The core re-architecture "Brain Transplant" is **Complete**. The system has successfully transitioned from a monolithic `rag/app` structure to the modular "Client-Engine" topology.

### Component Audit

| Component | Target State | Current Status | Notes |
|-----------|--------------|----------------|-------|
| **DeepDOC Engine** | Quarantined | ✅ **Quarantined** | Logic consolidated in `rag/parsers/deepdoc/`. No external calls except via **Client**. |
| **DeepDOC Client** | `deepdoc_client.py` | ✅ **Implemented** | `rag/parsers/deepdoc_client.py` acts as the facade. |
| **Orchestration** | `router.py` | ✅ **Implemented** | `rag/orchestration/router.py` handles dispatching using Client abstraction. |
| **Templates** | `semantic.py` | ✅ **Exists** | `rag/templates/semantic.py` is present. |
| **Project Structure** | `rag/parsers/` | ✅ **Aligned** | Clients (`docling`, `mineru`, `deepdoc`, `tcadp`) strictly separated. |
| **Legacy Code** | `rag/app/` | ✅ **Eliminated** | `rag/app` folder has been deleted. |

### Deviations from Proposal
1.  **DeepDOC Location:** The proposal suggested moving `deepdoc/` to the project root.
    *   **Decision:** We kept it at `rag/parsers/deepdoc/` for easier import management but strictly enforced the "Client Facade" pattern to achieve logical isolation.

### Completed Tasks
- [x] **Refactor Format Parsers:** Merged split logic (OCR, Layout, Table) back into `rag/parsers/deepdoc/` classes.
- [x] **Client Abstraction:** Created `DeepDocParser` to mirror `DoclingParser` and `MinerUParser`.
- [x] **Router Update:** `router.py` now uses the generic `DeepDocParser` interface, removing direct coupling to specific DeepDoc implementations.
- [x] **Cleanup:** Deleted `rag/app/format_parsers.py` and `rag/parsers/rag_*.py` wrappers.

### Next Immediate Focus
- **Phase 5 (UI Integration):** Expose `Semantic` template in the frontend dropdown.
- **Migration:** Validate `header_path` metadata storage in real-world ingestion.