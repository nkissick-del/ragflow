## 2024-05-22 - Anti-pattern: Manual Offset Pagination for Bulk Fetch
**Learning:** The codebase contained methods that manually paginated results using `offset` and `limit` inside a loop to fetch *all* records. This results in O(N/batch_size) queries and increasing database load as the offset grows (O(N^2) complexity for scanning).
**Action:** Replace manual offset loops with a single query iterator when fetching all records is intended and the result set size is manageable (e.g., fetching IDs). Peewee's `select()` returns an iterator that efficiently streams results.

## 2024-05-24 - Frontend: Memoizing Table Columns
**Learning:** In React table implementations (like TanStack Table), re-creating the `columns` definition array on every render forces the table library to recalculate its internal state, potentially leading to unnecessary processing and child component re-renders.
**Action:** Always wrap column definitions in `useMemo` when using `useReactTable` or similar libraries, ensuring that the column structure remains referentially stable unless its dependencies actually change.

## 2025-05-26 - Frontend: Memoizing List Cards
**Learning:** List components like `ChatCard` and `SearchCard` were re-rendering unnecessarily when parent state (e.g., search input) changed. This is a common pattern in this codebase where list items are not memoized.
**Action:** Wrap expensive list items in `React.memo` to prevent re-renders when props are stable.
