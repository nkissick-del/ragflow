## 2024-05-22 - Anti-pattern: Manual Offset Pagination for Bulk Fetch
**Learning:** The codebase contained methods that manually paginated results using `offset` and `limit` inside a loop to fetch *all* records. This results in O(N/batch_size) queries and increasing database load as the offset grows (O(N^2) complexity for scanning).
**Action:** Replace manual offset loops with a single query iterator when fetching all records is intended and the result set size is manageable (e.g., fetching IDs). Peewee's `select()` returns an iterator that efficiently streams results.

## 2024-05-24 - Frontend: Memoizing Table Columns
**Learning:** In React table implementations (like TanStack Table), re-creating the `columns` definition array on every render forces the table library to recalculate its internal state, potentially leading to unnecessary processing and child component re-renders.
**Action:** Always wrap column definitions in `useMemo` when using `useReactTable` or similar libraries, ensuring that the column structure remains referentially stable unless its dependencies actually change.

## 2024-05-25 - Frontend: Memoizing List Items with Parent State Changes
**Learning:** List components (like cards in a grid) are often rendered inside parents that manage frequent state updates (e.g., search input). Without memoization, every keystroke causes all list items to re-render, even if the list data is stable. Additionally, passing inline arrow functions as props (e.g., `onClick={() => handler(item)}`) defeats `React.memo` by creating new function references on every render.
**Action:** Wrap list item components in `React.memo` and ensure that callbacks passed to them are referentially stable (e.g., pass the handler function directly and let the child component invoke it with the necessary data, or use `useCallback`).
