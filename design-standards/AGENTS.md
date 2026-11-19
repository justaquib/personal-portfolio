# Site UI Standards

Use these standards when building or updating pages in this project.

## Visual language

- Use a calm, restrained palette: warm off-white page backgrounds, white surfaces, dark charcoal text, muted gray secondary text, and sage or olive green accents.
- Avoid saturated gradients, heavy shadows, and unrelated accent colors. Reserve strong color for meaningful status feedback.
- Use clear typographic hierarchy: bold, compact headings; readable body copy; small uppercase labels only for metadata.
- Prefer generous but consistent spacing. Keep controls and supporting content compact so the primary task stays prominent.
- Use subtle borders, rounded corners, and consistent hover and focus states.

## Layout and scrolling

- **Layouts must not be page-scrollable.** Fit the primary experience to the viewport whenever the page is a focused tool or prototype.
- Use viewport-aware sizing (`100dvh`), `min-height: 0`, and flexible regions so content fits without overflow.
- Avoid fixed heights that make content overflow on smaller screens. Reduce spacing and canvas height responsively.
- Do not add nested scroll areas as a workaround. If content cannot fit, reorganize or compact the layout while keeping controls and essential information visible.
- Prevent horizontal overflow at all viewport widths.
- Keep mobile navigation and actions accessible without requiring page scrolling.

## Interaction and accessibility

- Preserve existing application behavior while changing presentation.
- Make primary actions visually distinct and label icon-only actions accessibly.
- Keep text contrast readable and provide visible keyboard focus states.
- Ensure pointer and touch interactions work for drawing and drag controls.
