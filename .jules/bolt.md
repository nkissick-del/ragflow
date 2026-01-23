## 2024-05-22 - Anti-pattern: Manual Offset Pagination for Bulk Fetch
**Learning:** The codebase contained methods that manually paginated results using `offset` and `limit` inside a loop to fetch *all* records. This results in O(N/batch_size) queries and increasing database load as the offset grows (O(N^2) complexity for scanning).
**Action:** Replace manual offset loops with a single query iterator when fetching all records is intended and the result set size is manageable (e.g., fetching IDs). Peewee's `select()` returns an iterator that efficiently streams results.

## 2024-05-24 - Frontend: Memoizing Table Columns
**Learning:** In React table implementations (like TanStack Table), re-creating the `columns` definition array on every render forces the table library to recalculate its internal state, potentially leading to unnecessary processing and child component re-renders.
**Action:** Always wrap column definitions in `useMemo` when using `useReactTable` or similar libraries, ensuring that the column structure remains referentially stable unless its dependencies actually change.

## 2025-02-18 - Anti-pattern: Inline Regex Compilation in Loops
**Learning:** `rag/nlp/term_weight.py` contained multiple methods (`pretoken`, `weights`) that defined and compiled regular expressions inside loops or frequently called methods. This causes significant overhead (benchmarked ~3x slowdown).
**Action:** Move all regex definitions to module-level constants using `re.compile()` to ensure they are compiled once. This applies to any text processing modules in `rag/nlp/`.
