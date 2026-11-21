# Short note – what I did and why (Task 2)

## What I built

1. **JSON schema** for logistics actions (`type`, `entity`, `details`, `confidence`, `source_span`) plus a top-level wrapper that includes `needs_manual_review` and `review_reason`.
2. **Few-shot system prompt** with three concrete examples (schedule pickup, report delay, empty check-in) so the model consistently maps free-form driver speech onto the schema.
3. **OpenAI function-calling implementation** (`extract_logistics_actions`) using `gpt-4o-mini`, `strict: true`, and forced tool choice. The call is wrapped with retry logic for rate-limit / transient API errors.
4. **Fallback / low-confidence path**:
   - Any action whose confidence is below 0.65 is marked `needs_review=True`.
   - The whole result also carries a top-level `needs_manual_review` flag (plus a human-readable reason).
   - When no API key is present a deterministic mock extractor is used so tests and demos still work offline.
5. **Pytest suite** containing the three required synthetic transcripts that assert:
   - correct object shape (`type` / `entity` / `details`),
   - review flag behaviour on empty / low-confidence input,
   - JSON-serialisable output.

A small Streamlit app (`app.py`) demonstrates the required “collapsible panel + warning badge” UI.

## Why these choices

- **Function calling + strict mode** guarantees the shape the rest of the pipeline (confirmation UI, downstream task systems) can rely on, which is far safer than free-form JSON mode for an operations tool.
- **Explicit confidence + review flag** satisfies the mentor checklist item “when confidence is low, return a flag for manual review” and gives coordinators a clear signal that human eyes are needed.
- **Offline mock** keeps the unit tests green in CI without requiring secrets or network access, while still exercising the same public API surface that the live path uses.
- **Few-shot examples inside the system prompt** improve reliability on short, noisy voice-note language without needing fine-tuning.

This module is intentionally self-contained so Task 3 (confirmation UI) can simply call `extract_actions(transcript)` and render the returned objects.
