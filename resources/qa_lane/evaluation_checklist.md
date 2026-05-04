# QA Evaluation Checklist for AI Features

Use this checklist for every example from Day 2 onward.

## Prompt and input validation

- Is the prompt explicit about format and constraints?
- Are edge-case inputs included in tests?
- Is fallback behavior defined for missing API keys or context?

## Output quality checks

- Is output grounded in provided context (for RAG)?
- Are claims verifiable and non-contradictory?
- Does output follow requested structure (bullets/table/JSON)?

## Safety and reliability checks

- Does the app handle tool/API failure gracefully?
- Is timeout/retry behavior documented for external calls?
- Is sensitive data excluded from logs/reports?

## Regression checks

- Keep a fixed prompt set and compare outputs over time.
- Track changes in latency and token/cost estimates.
- Record known failure cases and re-test after every major change.

