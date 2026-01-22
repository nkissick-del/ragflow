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

## 3. The Vision: Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 1: PARSERS (Pre-Processing)                                          │
│  "Make the document machine-readable"                                       │
│                                                                             │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐                │
│  │  Docling  │  │  MinerU   │  │ DeepDOC   │  │  Future   │                │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘                │
│        │              │              │              │                       │
│        └──────────────┴──────────────┴──────────────┘                       │
│                              │                                              │
│                    (Parser-specific output)                                 │
└──────────────────────────────┼──────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 2: ORCHESTRATION (Normalization)                                     │
│  "Translate parser outputs into standardized IR"                            │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Parser Adapters                                                     │   │
│  │  ┌────────────────┐ ┌────────────────┐ ┌────────────────┐           │   │
│  │  │ DoclingAdapter │ │ MinerUAdapter  │ │ DeepDOCAdapter │           │   │
│  │  └───────┬────────┘ └───────┬────────┘ └───────┬────────┘           │   │
│  │          └──────────────────┴──────────────────┘                     │   │
│  │                             │                                        │   │
│  │                             ▼                                        │   │
│  │              ┌───────────────────────────────┐                       │   │
│  │              │   STANDARDIZED IR (Contract)  │                       │   │
│  │              │   - Structured Markdown       │                       │   │
│  │              │   - Header hierarchy metadata │                       │   │
│  │              │   - Table/code block markers  │                       │   │
│  │              └───────────────────────────────┘                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────┼──────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 3: TEMPLATES (Business Logic)                                        │
│  "Optimize chunking for retrieval based on domain knowledge"                │
│                                                                             │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐                │
│  │  General  │  │ AEMO ISP  │  │   Legal   │  │  Future   │                │
│  │(Semantic) │  │(Regulatory)│ │(Contracts)│  │           │                │
│  └───────────┘  └───────────┘  └───────────┘  └───────────┘                │
│                                                                             │
│  Templates are PARSER-AGNOSTIC. They consume the Standardized IR.          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Layer Responsibilities

### Layer 1: Parsers
| Attribute | Value |
|-----------|-------|
| **Role** | "Solve the File Format" |
| **Input** | Raw PDF, DOCX, HTML, etc. |
| **Components** | `docling_parser.py`, `mineru_parser.py`, `deepdoc_parser.py` |
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
    
    # The full structured content (Markdown with headers, tables, code blocks)
    content: str
    
    # Metadata about the document
    metadata: Dict[str, Any] = field(default_factory=dict)  # {"source": "docling", "pages": 10, ...}
    
    # Pre-parsed structure (optional, for efficiency)
    elements: Optional[List[DocumentElement]] = None  # headers, paragraphs, tables, code blocks
```

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

### Implementation Phases

- [ ] **Phase 1:** Modify `docling_parser.py` to return structured Markdown
- [ ] **Phase 2:** Add `DoclingAdapter` to orchestration layer
- [ ] **Phase 3:** Create `templates/semantic.py` with header tracking
- [ ] **Phase 4:** Update routing logic for structured parsers
- [ ] **Phase 5:** Expose "Semantic" template in UI dropdown

### Testing Requirements

> [!IMPORTANT]
> Each phase must include testing before proceeding to the next phase.

| Phase | Testing Approach |
|-------|------------------|
| **Phase 1** | Unit test: Verify `docling_parser.py` returns structured Markdown (not `splitlines()`) |
| **Phase 2** | Unit test: Verify `DoclingAdapter` converts Docling output → Standardized IR |
| **Phase 3** | Integration test: End-to-end test with sample PDF → verify chunks have `header_path` metadata |
| **Phase 4** | Integration test: Verify routing logic correctly selects semantic template for Docling jobs |
| **E2E Validation** | Manual test: Upload a structured PDF via RAGFlow UI, verify chunks in database have correct header hierarchy |

**Existing Test Location:** `test/integration/test_docling_integration.py` — extend this for new pipeline tests.