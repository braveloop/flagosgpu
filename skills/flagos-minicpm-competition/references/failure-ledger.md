# Competition failure ledger

Read this ledger before diagnosing a repeated build, runtime, correctness, accuracy, performance, or platform failure. Add an entry once the same blocker has survived three evidence-backed attempts, then consult the official-source chain before another implementation attempt. Never record passwords, tokens, private registry credentials, or other secrets.

An attempt is one distinct hypothesis with a scoped diagnostic or change and an observable result. Re-running the same command without a changed hypothesis is a repetition, not a new attempt.

| Date | Platform / stage | Symptom and stable reproduction | Attempt 1 | Attempt 2 | Attempt 3 | Official evidence consulted | Revised model / next action | Verification / artifacts |
|---|---|---|---|---|---|---|---|---|
| — | — | No three-attempt blocker recorded yet | — | — | — | — | — | — |

After resolution, replace tentative language with the verified root cause and final fix. Keep the failed evidence and passing artifact identifiers so a future agent can distinguish recurrence from a superficially similar symptom.
