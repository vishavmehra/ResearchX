# Laya evaluation

laya-eval-0.1.0 · laya-decisions-0.2.0 · service ready · mode shadow

## Production traces (shadow mode)

649 laya_decision rows · status {'skipped': 637, 'completed': 11, 'failed': 1} · latency p50 / p95 1211 / 2255 ms · workflow agreement with the rules 0.889 (8/9) · injection agreement 1.0 (11/11) · decisions applied 0

## Labelled prompts

30 prompts (14 injection, 10 research, 6 hard negatives) · 30 answered · latency p50 / p95 1195 / 1349 ms

| Detector | Injection detection |
|---|---|
| Laya (injection >= 0.5) | P 1.0 · R 0.857 · F1 0.923 (tp 12, fp 0, fn 2) |
| Deterministic pattern | P 0.667 · R 0.143 · F1 0.235 (tp 2, fp 1, fn 12) |
| Either | P 0.923 · R 0.857 · F1 0.889 (tp 12, fp 1, fn 2) |

Hard negatives flagged: Laya 0, pattern 1 of 6. Workflow agreement with the rules on research questions: 0.889 (8/9).

The labels are the author's, on a small hand-written set: a sanity check, not a measure of attack coverage. Laya stays advisory: it never blocks a request or overrides a rule.
