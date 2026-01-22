## 2026-01-17 - Uncovering Hidden Accessibility Features
**Learning:** I discovered that developers sometimes comment out accessibility features (like `aria-label`) because they lack localization support or are unsure about the implementation.
**Action:** Always check commented-out code in UI components for accessibility features that can be easily re-enabled, even if it requires a hardcoded fallback or a small workaround.

## 2026-01-17 - Focus Styles in Tailwind
**Learning:** Custom components that override background colors (e.g., `bg-bg-card`) can inadvertently hide default focus indicators if they rely on background color changes.
**Action:** When overriding background colors on interactive elements, always explicitly define `focus-visible` styles (like `ring-2`) to ensuring keyboard navigability is maintained.
## 2024-05-23 - Composing Tooltips with Dialog Triggers
**Learning:** Adding a Tooltip to a Button that is already a trigger for a Radix Dialog (via `ConfirmDeleteDialog`) requires careful nesting. The `Tooltip` must wrap the `ConfirmDeleteDialog`, and the `ConfirmDeleteDialog` must wrap the `TooltipTrigger`. The `TooltipTrigger` then wraps the `Button` using `asChild`. This ensures that props/refs are correctly forwarded from both the Dialog system and the Tooltip system to the underlying Button.
**Action:** Use this composition pattern whenever adding tooltips to existing action buttons that trigger dialogs.

## 2025-05-24 - Manual Accessibility in AlertDialog
**Learning:** Unlike the standard `Dialog` component which often includes an accessible close button by default (via `DialogPrimitive.Close`), the `AlertDialog` component (wrapping `AlertDialogPrimitive`) does not automatically include a close button. When developers manually add a close button (e.g., using `AlertDialogCancel` with an icon), they must explicitly add `aria-label` or `sr-only` text, as it's not inherited or auto-generated.
**Action:** Always verify that manual "X" close buttons in `AlertDialog` components have accessible labels, especially when they only contain icons.
