# CargoLink Logistics – Voice Note Assistant (Task 2)

**LLM-driven action extraction with function calling**

Part of the Ezitech AI Internship project *Voice Note Assistant for CargoLink Logistics*.

## What this task delivers

- A JSON schema for logistics actions (`type`, `entity`, `details`, `confidence`, …)
- A few-shot system prompt that maps voice-note transcripts → structured actions
- OpenAI function-calling implementation (`gpt-4o-mini` by default) with `strict: true`
- Low-confidence / ambiguous fallback that sets a `needs_manual_review` flag
- Pytest suite with three synthetic transcripts verifying correct JSON output
- Streamlit demo that shows a collapsible JSON panel + warning badges for low-confidence items

## Quick start

```bash
# 1. Clone / enter the repo
cd cargolink-voice-assistant-task2

# 2. Create virtualenv & install
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. (Optional) add your key for live calls
export OPENAI_API_KEY=sk-...

# 4. Run unit tests (works offline – uses mock extractor)
pytest tests/ -v

# 5. Launch the Streamlit demo
streamlit run app.py
```

## Project layout

```
├── app.py                     # Streamlit UI (collapsible panel + badges)
├── requirements.txt
├── README.md
├── src/
│   └── action_extractor.py    # Core extraction logic + schema + prompt
└── tests/
    └── test_action_extractor.py
```

## Extraction contract

```python
result = extract_actions(transcript)

# result.actions          → List[LogisticsAction]
# result.needs_manual_review → bool
# result.review_reason    → Optional[str]
# result.to_dict()        → JSON-serializable dict
```

Each `LogisticsAction` contains:

| Field        | Type          | Description                                      |
|--------------|---------------|--------------------------------------------------|
| `type`       | str (enum)    | schedule_pickup, update_status, report_delay, …  |
| `entity`     | str           | Shipment ID, truck, driver, customer, location   |
| `details`    | dict          | location, time, status, reason, priority, notes  |
| `confidence` | float 0–1     | Model confidence                                 |
| `needs_review` | bool        | True when confidence < 0.65                      |
| `source_span`| str \| None   | Supporting quote from the transcript             |

## Design decisions

1. **Strict function calling** – The tool definition uses `"strict": true` and `additionalProperties: false` so the model is forced to emit schema-conformant JSON.
2. **Confidence threshold** – Any action with `confidence < 0.65` (or an empty action list) flips the top-level `needs_manual_review` flag. This is the required fallback for low-confidence responses.
3. **Offline mock** – When `OPENAI_API_KEY` is absent the extractor falls back to a deterministic keyword heuristic. This keeps CI and local pytest green without spending tokens.
4. **Few-shot prompt** – Three short examples (pickup, delay, empty check-in) are embedded in the system prompt to steer the model toward the expected schema.

## Mentor checklist coverage

- [x] Extraction function returns a list of action objects with `type`, `entity`, and `details`
- [x] When confidence is low, the function returns a flag for manual review
- [x] Unit test with three sample transcripts verifies correct JSON output

## Next steps (later tasks)

- Task 3 – Confirmation UI for editing / approving the extracted actions
- Task 4 – Accuracy evaluation dashboard with WER
- Task 5 – Secure CI/CD to Streamlit Community Cloud + hand-over package

---

*Intern: Waqar Nadeem · Technology: Artificial Intelligence · Level: Advanced*
