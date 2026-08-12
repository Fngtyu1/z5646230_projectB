# Prompt log — app, report, and hand-in QA

## What I wanted

Turn the generated evidence into a usable investor journey, complete report,
transparent workflow pack, and reproducible hand-in folder.

## Prompt(s)

“Build a lightweight Streamlit app from precomputed artifacts only; include fund
comparison, each fund's fact sheet, allocation controls, sentiment, innovation
and limitations. Draft DOCX/PDF, inspect every page, run check_handin, and leave
clear GitHub/Streamlit steps for me.”

## What the assistant produced

A five-tab MarketLens dashboard, editable Word report, PDF, deployment handoff,
prompt logs, tests and submission checklist.

## What was wrong or risky

The first portfolio-allocation exhibit had a crowded legend and the first
risk/return map had overlapping labels. AI-drafted report prose also cannot
substitute for the student's own interpretation.

## What I changed and why

I required the two figures to be regenerated and visually rechecked, tested the
app with Streamlit's testing interface, and created a student review guide that
identifies the conclusions requiring personal rewriting or approval. Deployment
is deliberately left for the student because it requires authenticated accounts.
