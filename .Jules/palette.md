## 2026-01-17 - Uncovering Hidden Accessibility Features
**Learning:** I discovered that developers sometimes comment out accessibility features (like `aria-label`) because they lack localization support or are unsure about the implementation.
**Action:** Always check commented-out code in UI components for accessibility features that can be easily re-enabled, even if it requires a hardcoded fallback or a small workaround.

## 2026-01-17 - Focus Styles in Tailwind
**Learning:** Custom components that override background colors (e.g., `bg-bg-card`) can inadvertently hide default focus indicators if they rely on background color changes.
**Action:** When overriding background colors on interactive elements, always explicitly define `focus-visible` styles (like `ring-2`) to ensuring keyboard navigability is maintained.
## 2024-05-23 - Composing Tooltips with Dialog Triggers
**Learning:** Adding a Tooltip to a Button that is already a trigger for a Radix Dialog (via `ConfirmDeleteDialog`) requires careful nesting. The `Tooltip` must wrap the `ConfirmDeleteDialog`, and the `ConfirmDeleteDialog` must wrap the `TooltipTrigger`. The `TooltipTrigger` then wraps the `Button` using `asChild`. This ensures that props/refs are correctly forwarded from both the Dialog system and the Tooltip system to the underlying Button.
**Action:** Use this composition pattern whenever adding tooltips to existing action buttons that trigger dialogs.

## 2026-01-23 - Avoiding Ghost Clicks on Accessible Hover Elements
**Learning:** Replacing `invisible` with `opacity-0` for keyboard accessibility (so elements remain in the tab order) has a side effect: the element remains clickable even when visually hidden ("ghost clicks").
**Action:** Always pair `opacity-0` with `pointer-events-none`, and restore both `opacity-100` and `pointer-events-auto` on `focus-visible` and `group-hover`. This ensures the element is only interactive when it is perceivable.

## 2026-02-27 - Manual Close Buttons in Dialogs
**Learning:** `ConfirmDeleteDialog` manually implements a close button using `AlertDialogCancel` with an icon-only `X`. This bypasses default accessibility features if developers forget to add `aria-label`.
**Action:** When componentizing dialogs with custom close buttons, explicitly check that icon-only buttons have localized `aria-label`s.
