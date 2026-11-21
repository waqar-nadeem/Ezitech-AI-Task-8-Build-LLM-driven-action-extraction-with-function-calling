"""
Streamlit demo for Task 2 – LLM-driven action extraction.
Shows a collapsible panel with extracted actions as JSON and warning badges
for low-confidence items.
"""

import json
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from action_extractor import extract_actions, ActionExtractor

st.set_page_config(
    page_title="CargoLink – Action Extraction",
    page_icon="📦",
    layout="wide",
)

st.title("📦 CargoLink Logistics – Voice Note Action Extraction")
st.caption("Task 2 · LLM-driven extraction with function calling & low-confidence fallback")

# Sidebar
with st.sidebar:
    st.header("Settings")
    use_live = st.checkbox(
        "Use live OpenAI API (requires OPENAI_API_KEY)",
        value=bool(os.getenv("OPENAI_API_KEY")),
    )
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"], index=0)
    st.markdown("---")
    st.markdown(
        "**How it works**\n\n"
        "1. Paste a transcript (or use a sample).\n"
        "2. The extractor calls `extract_logistics_actions` via function calling.\n"
        "3. Low-confidence items are flagged for manual review."
    )

# Sample transcripts
SAMPLES = {
    "Schedule pickup": (
        "Hey dispatch, can we schedule a pickup for shipment CL-7842 at the "
        "downtown warehouse tomorrow morning around 9 AM? Priority is high."
    ),
    "Report delay": (
        "This is driver of Truck 12. We are stuck on the highway due to a major "
        "accident. We will be at least 45 minutes late for the delivery to client Apex."
    ),
    "No action / check-in": (
        "Just checking in. Nothing special today, traffic is normal, all good."
    ),
    "Mixed / ambiguous": (
        "Um, yeah, I think the load might be ready, not sure about the time though, "
        "maybe call the warehouse later?"
    ),
}

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Transcript")
    sample_name = st.selectbox("Load sample", list(SAMPLES.keys()))
    if st.button("Load selected sample"):
        st.session_state["transcript"] = SAMPLES[sample_name]

    transcript = st.text_area(
        "Paste or edit the voice-note transcript",
        value=st.session_state.get("transcript", SAMPLES["Schedule pickup"]),
        height=220,
    )

    run = st.button("Extract actions", type="primary")

with col2:
    st.subheader("Extracted Actions")

    if run or "last_result" in st.session_state:
        if run:
            kwargs = {}
            if not use_live:
                # Force mock path
                os.environ.pop("OPENAI_API_KEY", None)
            result = extract_actions(transcript, model=model)
            st.session_state["last_result"] = result
        else:
            result = st.session_state["last_result"]

        # Overall review badge
        if result.needs_manual_review:
            st.warning(
                f"⚠️ Manual review recommended"
                + (f": {result.review_reason}" if result.review_reason else "")
            )
        else:
            st.success("✅ Extraction confidence looks good")

        if result.error:
            st.error(f"Error: {result.error}")

        # Collapsible JSON panel
        with st.expander("📋 Actions (JSON)", expanded=True):
            st.json(result.to_dict())

        # Pretty cards
        if result.actions:
            for i, action in enumerate(result.actions, 1):
                badge = "🔴 LOW CONFIDENCE" if action.needs_review else "🟢 OK"
                with st.container(border=True):
                    st.markdown(f"**Action {i}** · `{action.type}` · {badge}")
                    st.markdown(f"**Entity:** {action.entity}")
                    st.markdown(f"**Confidence:** {action.confidence:.2f}")
                    if action.details:
                        st.markdown("**Details:**")
                        st.json(action.details)
                    if action.source_span:
                        st.caption(f"Source: “{action.source_span}”")
        else:
            st.info("No actions extracted.")

st.markdown("---")
st.caption(
    "CargoLink Logistics · Ezitech AI Internship · Task 2 – Build LLM-driven action extraction with function calling"
)
