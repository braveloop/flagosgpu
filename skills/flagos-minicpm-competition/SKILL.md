---
name: flagos-minicpm-competition
description: Use for every planning, implementation, review, debugging, profiling, benchmarking, accuracy-evaluation, documentation, and submission task in /home/brave/flagos for the FlagOS Season 2 MiniCPM5-2B operator optimization competition on Iluvatar BI-V150 and MetaX C500. Enforces contest rules, official-source research, two-model planning/execution routing, no local GPU execution, staged optimization gates, and escalation after three failed attempts.
metadata:
  short-description: FlagOS MiniCPM5-2B competition engineering rules
---

# FlagOS MiniCPM Competition Engineering

Use this Skill for all work under `/home/brave/flagos`. The objective is a reproducible, rule-compliant MiniCPM5-2B throughput improvement on both official accelerator platforms, not an isolated microbenchmark win.

## Highest-priority constraints

- Treat the [competition page](https://flagos.io/race-detail-season2?id=539vlt2p&lang=cn) as the highest authority. Re-read it before baseline collection, before any rule-sensitive optimization, and before final submission because web rules may change.
- Use the local NVIDIA environment as the source of truth for source and documentation edits, static review, Git operations, and synchronization to remote compute. Do not install project dependencies, compile, import project modules, start services, load the model, execute tests, profile, or benchmark on this host.
- Perform development-environment setup, dependency installation, compilation, imports, and operator, service, model, memory, compiler, device, and performance diagnosis on rented target compute. A rented 16/32 GB instance or an unapproved 64 GB instance supports diagnosis and prevalidation only; its results are not official competition evidence.
- Run formal accuracy evaluation, 4K/1K and 16K/1K baselines, final scoring, and submission evidence only on organizer-accepted BI-V150 and C500 64 GB environments. Lack of official access does not block rented-platform diagnosis, but formal verification remains pending until the official environments are available; never infer an official pass from local static review or rented-platform results.
- Preserve the official benchmark/test cases, service arguments, prompts, sampling behavior, model semantics, output lengths, and scoring inputs. Do not obtain a score by changing the workload.
- Do not switch to a faster existing backend, merge or cherry-pick a newer framework implementation, enable existing quantization/speculative decoding/prefix-cache settings, or encode `4096`/`16384` benchmark-specific branches. A performance submission must contain substantive implementation work and a general shape-aware policy.
- Keep user changes intact. Inspect the relevant repository status before editing, touch only task-owned files, and never use destructive Git commands.

Read [`references/official-resources.md`](references/official-resources.md) before material design or debugging work. Read [`references/failure-ledger.md`](references/failure-ledger.md) before diagnosing a repeated failure.

## Required model routing

When agent spawning with model overrides is available, route every repository mutation through these two sequential stages. This is an actual handoff between agents; never claim that the current model changed itself.

1. Spawn a read-only planning agent using model `gpt-6-astra` with reasoning effort `xhigh`. It must inspect current files, the competition rules, the relevant FlagOS official material, the corresponding vendor material when platform behavior is involved, and the current Git state. Its output must define the smallest change, affected platform, invariants, risk, rollback, and exact remote diagnostic and official-platform acceptance plans.
2. After accepting the plan, spawn an execution agent using model `gpt-5.6-sol` with reasoning effort `high`. Give it the plan, exact paths and commits, user constraints, platform boundary, official references, and available test access. This agent implements the change and performs only the verification permitted by the execution-location rules.
3. The primary agent performs a read-only final diff and evidence audit. Any material corrective edit discovered in audit requires a fresh `gpt-6-astra` xhigh planning pass followed by a `gpt-5.6-sol` high execution pass.

Use the canonical identifier `gpt-5.6-sol`; it is the normalized form of the requested “GPT-5.6 sol” execution model. Use a bounded context fork or no fork when a model override rejects full-history forking, and include all necessary context explicitly. Run planner and executor sequentially.

Do not silently substitute models or reasoning effort. If spawning, a required model, planning `xhigh` effort, or execution `high` effort is unavailable, stop before mutation and report the failed routing capability. Purely read-only explanation and status work may continue without the two-agent route.

## Competition and model invariants

The locally frozen source set is:

| Component | Required revision | Role |
|---|---|---|
| `vllm-plugin-FL` | branch `flagos-2026-s2`, commit `13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0` | Primary framework and platform integration worktree |
| `FlagGems` | tag `v5.3.5`, commit `a7620cc191a0b42e040194622c5758b22a7a25dc` | Primary Triton/operator worktree |
| `vllm` | tag `v0.24.0`, commit `ee0da84ab9e04ac7610e28580af62c365e898389` | Read-only upstream reference |

## GitHub workflow (project convention)

- Develop in `bravegpuwinner/s2-dev` in both `vllm-plugin-FL` and `FlagGems`, created from their fixed official commits above. Preserve the official branch and tag; keep `vllm` read-only. This branch choice is a project convention, not an organizer requirement.
- Treat local source files and Git history as the source of truth. Keep `origin` pointed at the official repositories. After the user's GitHub fork URLs are confirmed, add separate fork remotes and push explicitly only to those forks.
- On remote compute, start from a fixed base commit and verify both the checked-out source and the code actually imported by the running environment. A committed experiment identity includes the repository, full commit SHA, actual remote import path, and immutable experiment ID. An uncommitted experiment identity instead includes the repository, full base SHA, SHA-256 of the complete archived scoped diff, SHA-256 hashes of every changed or new file, actual remote import path, and immutable experiment ID; a base commit alone never identifies a dirty run. Keep the clean-worktree guard in `scripts/c500_persistent_workspace.sh`; dirty experiments require a separate, explicit manifest and activation workflow rather than implicit synchronization.
- Batch related edits and experiments until they form a coherent stage deliverable with the necessary validation and a retain/continue/rollback decision, then create a scoped commit and push when appropriate. An explicit user request may authorize an earlier commit, but the request to initialize and push the root repository authorizes only that one commit and is not recurring permission. Do not commit every tiny edit, candidate experiment, failed run, documentation update, or synchronization step. Preserve already-pushed history, including `47882ba`; never rewrite or force-push it. Retain only evidence-backed patches in the final implementation: platform-specific optimizations must demonstrate benefit on their target platform and pass regression checks on the other platform without degradation. Integrate the accepted patches and reproduce the combined result on both official platforms before final submission.

Record the actual official-platform import paths and installed commits before trusting these local pins. Do not update a branch or tag merely because upstream moved.

MiniCPM5-2B is a dense Llama-family BF16 model with 42 layers, hidden size 2048, intermediate size 6144, 16 query heads, 2 KV heads, head dimension 128, and a 131072-token context limit. Its GQA ratio is 8:1. BF16 KV cache costs approximately `42 × 2(K/V) × 2(KV heads) × 128 × 2 bytes = 43008 bytes`, about 42 KiB per token; 64 concurrent 16K-input plus 1K-output requests approach 44.6 GiB before weights, activations, workspaces, graph buffers, and allocator reserve.

MATH-500 Level 3 contains 105 questions. A baseline around 0.962 is about 101/105 while the `>=0.95` gate requires at least 100/105, leaving roughly one additional error. Treat accuracy as a hard gate after every numerical-path change; W4 and KV-cache quantization are high-risk and are not early-stage options.

The pinned local competition branch config routes C500 Attention through the MetaX/vendor path. Its BI-V150 config gives the FlagGems selector priority, but that selector returns the `TRITON_ATTN` enum path unless `VLLM_FL_USE_FLAGGEMS_ATTN` is explicitly enabled; the official image may also differ through environment variables, registration overrides, or patches. Therefore, freeze the actual backend from the unmodified official serve command, whitelisted environment evidence, and server logs before assigning BI-V150 candidates. Apply C0 and the FlagGems Attention candidates only when the target implementation is actually active; otherwise close C0 as not applicable and re-profile the real backend. Never change the backend merely to make a planned candidate apply. Share interfaces, evidence formats, and general scheduling logic where sensible, but keep chip-specific kernels and heuristics separate.

The current plan records four official scenarios: BI-V150 and C500, each with 4K/1K and 16K/1K workloads. Optimize both lengths. Preserve Mean TTFT within the organizer-confirmed limit and require throughput gains to exceed measured run-to-run noise. Until clarified in writing, do not assume how the two platform scores form the stated 70% performance component, whether the TTFT 1% limit is per run/per scenario/aggregate, or how `temperature=1.0` seed and repetition are judged.

Treat Season 1 expert feedback only as historical method guidance, never as a Season 2 rule or permission source. An OOT wrapper, dispatch or model-runner integration, a switch to an existing vendor/native/Inductor implementation, or merely enabling an existing capability is not by itself an innovation. A retained performance strategy needs substantive implementation work in the final diff and an exact report-to-code mapping.

## Official-source workflow

Use primary sources in this order: current competition page and written organizer answers; FlagOS documentation, SkillHub, `kernelgen-flagos`, and pinned FlagOS repositories; the official Iluvatar or MetaX developer material for platform behavior; upstream project documentation and authoritative GitHub source/issues/PRs. Use blogs or other secondary sources only when primary evidence is unavailable, and label them as secondary.

Before generating or optimizing a Triton operator, inspect the current official [`kernelgen-flagos`](https://flagos.io/SkillHubDetail?skillName=kernelgen-flagos) workflow and its [open-source Skill](https://github.com/flagos-ai/skills/tree/main/skills/kernelgen-flagos). It is a mandatory workflow reference. If that Skill is actually invoked, obey its repository detection, full sub-skill reading, MCP setup, generation/optimization, testing, and feedback rules; do not pretend that KernelGen or its MCP was used when it was unavailable.

For each material design decision, record the source URL or local commit, the exact idea adopted, the platform/version boundary, and what was intentionally not copied. Official material informs the implementation but never overrides a more restrictive competition rule.

An “attempt” means a distinct, evidence-backed hypothesis followed by one scoped change or diagnostic and a defined verification result. Random retries and parameter nudges do not count as new attempts. If the same problem remains unsolved after three attempts:

1. Stop speculative local changes; do not begin a fourth variation.
2. Add the three attempts and current evidence to `references/failure-ledger.md`.
3. Re-check FlagOS docs, SkillHub, `kernelgen-flagos`, relevant official repository history/issues, and the corresponding Iluvatar or MetaX documentation.
4. Reformulate the root-cause model from that evidence. Continue only with a materially new authoritative lead; otherwise report the blocker and the precise missing information.

## Staged optimization workflow

Read [`DETAILED_OPTIMIZATION_PLAN.md`](../../DETAILED_OPTIMIZATION_PLAN.md) completely before performance work and use its seven phases in order. Do not enter a later phase until the earlier phase’s acceptance gate is met:

1. Clarify rules, fingerprint both environments, freeze source identity, establish correct output and repeatable official baselines.
2. Profile 4K and 16K on both platforms and rank bottlenecks with end-to-end evidence.
3. Remove C500 synchronization, temporary allocation, and wrapper overhead without changing semantics.
4. Optimize the BI-V150 backend proven active by official logs; use the planned FlagGems GQA Decode, long-Prefill Attention, Paged KV, and KV-write candidates only if their target implementation is active.
5. Optimize the remaining C500 Attention path and shared RoPE, RMSNorm, SiLU-gate, QKV/KV-write fusion opportunities.
6. Evaluate general shape-aware scheduling and graph coverage; consider self-developed W8A16 only after written rule confirmation and evidence that it targets the dominant bottleneck.
7. Combine platform-specific patches, run ablations and clean reproductions, remove unproven code, and make the report-to-diff mapping exact.

Each patch must test one performance hypothesis. Start from trace evidence, define expected counter movement, preserve a fallback for unsupported shapes/layouts/dtypes, and delete a patch that wins only a microbenchmark or falls within noise without enabling a required later change.

Favor Attention and KV-cache work first because long-context memory pressure is dominant. After BI-V150 backend identity is proven, tune its applicable path against GQA 16Q/2KV, head dimension 128, BF16, paged layouts, mixed sequence lengths, and chunked prefill. For Decode, inspect launch-heavy repeated operations across 42 layers, including fused residual-add plus RMSNorm, fused SiLU times gate, RoPE/QKV post-processing/KV write, and graph breaks. Scheduler changes must be expressed in semantic quantities such as phase, active sequences, token budget, maximum KV length, memory pressure, and graph eligibility—not official case names or exact benchmark lengths.

Quantization is optional and last. Prefer self-developed weight-only INT8/W8A16 over W4A16 or KV quantization, require organizer approval first, and run the complete Level 3 evaluation immediately after every candidate. Delete the path on any accuracy, token behavior, unsupported-shape, or net end-to-end performance failure.

## Implementation and verification discipline

Before implementing a performance change that depends on an actual runtime path or backend, confirm on the target under diagnosis the complete chain from model class through registration and dispatch to the compiled implementation and actual device kernel. If the remote environment is not yet available, local static development and candidate preparation may proceed with runtime-path and backend assumptions explicitly marked as pending verification. Confirm the formal chain separately on the official platform before acceptance; a source symbol, OOT entrypoint, configuration file, or selected backend name alone does not prove activation. Search for existing helpers and configuration before adding parallel logic. Keep the hot path small, isolate vendor specialization under existing dispatch boundaries, retain correctness fallbacks, and remove dead experiments in the same change.

Edit source and documentation in the local NVIDIA source of truth, then synchronize the scoped changes to the remote target. Before trusting remote results, verify the exact commit, applied patch, and source hashes against the local source, as well as actual imported code paths and device, image, driver, software, and backend identities on that target. Return logs, measurements, and environment evidence under an immutable experiment ID for local archiving; check that evidence and remote state are captured before releasing rented compute.

Local verification is static only: inspect diffs, run `git diff --check` in touched repositories, verify links/paths, and review control/data flow without importing or executing project code. Do not describe static review as runtime validation.

Rented-platform diagnosis proceeds from environment and dependency checks through build/import checks, operator correctness, service/model smoke tests, profiling, and performance prevalidation. Fit diagnostic workloads to available memory and label results from 16/32 GB or unapproved 64 GB instances as preliminary; continue this diagnosis when official access is unavailable while keeping formal acceptance pending.

Official-platform verification proceeds from cheap to expensive:

- Differential operator tests across batch, sequence length, mixed lengths, layouts, boundary shapes, and fallback paths; compare against the original implementation with dtype-appropriate tolerances.
- Deterministic short-prompt and long/mixed-prompt smoke tests that check tokens, EOS behavior, generated length, request success, OOM, deadlock, and intermittent errors.
- Complete MATH-500 Level 3 with per-question output, exact sampling configuration, seed information, and final score.
- Official 4K/1K and 16K/1K benchmark commands on both platforms. Preserve every raw run; use the official repetition/statistics procedure, currently four runs with the first discarded and the final three summarized, unless the organizer gives a newer written rule.
- End-to-end paired controls and ablation with identical environment, service arguments, workload, thermal state, source commits, and startup mode. Report mean, standard deviation, coefficient of variation, total/output throughput, Mean/median/P99 TTFT, TPOT, ITL, successful requests, peak memory, logs, and profiler evidence.

Any incorrect output or failed request invalidates the associated performance result. A 4K gain cannot conceal a 16K regression, and a single best run is not evidence. Keep raw CSV, summary CSV, service logs, accuracy outputs, environment manifest, source hashes, trace/profile artifacts, and organizer approvals together under an immutable experiment identifier.

Attribute each candidate independently on the same platform, service parameters, source and dependency versions, backend, and warm-state procedure. Profile official shapes after warm-up, use the measured time share in the Amdahl bound, and retain internal paired controls in the order "unmodified original -> single candidate -> restored original retest" so drift can reject a false gain. Maintain ablation order and candidate-to-source-to-report mapping from the candidate's start. Never present an inactive path, no-code idea, existing-backend switch, or microbenchmark-only win as a performance result.

## Documentation routing and learning record

After every remote experiment finishes, promptly synchronize its raw outputs and exact source identity to the local project, verify the transferred results, and update the relevant existing learning document or active-plan status in the same work session before reporting the experiment complete. Replace obsolete summary values, run counts, code descriptions, and status in place instead of accumulating duplicate historical tables; retain history only when it explains the current conclusion. Distinguish an exact official two-scenario run from a single-scenario diagnostic repeat, and never promote either a rented-platform result or a microbenchmark to official acceptance. Do not create a new `SNN` document or a tiny Git commit merely because one run ended. If synchronization or documentation cannot finish, state what remains stale immediately rather than describing the work as complete.

Store the competition learning record in [`docs/`](../../docs/), and use [`docs/S00-学习索引.md`](../../docs/S00-学习索引.md) as its seven-stage index. Create a new globally numbered `SNN-中文主题.md` reproduction document only after a substantive stage deliverable is complete, reproducible, and has finished the necessary verification for its stated scope. A qualifying baseline result combines environment identity, a complete MATH-500 Level 3 run, repeated 4K/1K and 16K/1K runs, and a clear conclusion. A qualifying optimization result combines the implementation, correctness and fallback checks, a complete Level 3 run, both benchmark scenarios, and a verified retain or rollback decision. Merely making an environment path usable, identifying one root cause, preparing an unvalidated patch, completing one command or run, or deciding only to continue validation does not qualify for a new document. Never pre-create placeholders. Raw CSV, logs, manifests, per-question outputs, hashes, and traces remain grouped outside the learning prose by immutable experiment ID and do not count as learning documents. Rented-C500 results may document a fully verified diagnosis or prevalidation scope but must keep formal C500 and BI-V150 phase acceptance pending until organizer acceptance exists. Before creating or materially updating a reproduction document, read [`references/documentation-standard.md`](references/documentation-standard.md) completely and follow it; minor spelling, link, or status-only corrections require neither a new document nor another full read.

In learning prose, `A0` means only the organizer-published official baseline and `B1` means the current improvement. Unknown A0 fields stay `—`; never fill them from a rented or internal unmodified run. Public result summaries contain only A0/B1 for 4K/1K and 16K/1K. The internal paired controls above remain archived under immutable historical experiment IDs for causal checks and drift rejection, but are not a third public result class; do not rename historical IDs that already contain `A0`.

Write reproduction documents in Chinese for a zero-background reader, with a 150–300-character plain-language introduction, 4–6 meaningful main sections, connected explanatory paragraphs, and readable numbered implementation steps of 1–3 sentences each with a blank line between steps. Include the exact commands, working directory and environment-variable scope, hardware and software identity, repository SHAs and actual import paths, expected and actual results. Never include credentials. Learning prose must not cite experiment files, directories, links, hashes, raw CSV, logs, manifests, traces, or per-question outputs, and must not create standalone evidence, recovery/rollback, or remaining-boundary chapters. Keep those internal artifacts outside the learning prose while preserving numerical results, source symbols and commits, platform conditions, and exact official reproduction commands. Interleave source, code, reasoning, and results; define technical terms against exact symbols and pinned commits; state formula units and assumptions; and record the experiment identity needed to distinguish runs. Present BI-V150 and C500 separately, and within each platform distinguish 4K/16K and Prefill/Decode rather than blending them into one conclusion.

Follow the seven phases in [`DETAILED_OPTIMIZATION_PLAN.md`](../../DETAILED_OPTIMIZATION_PLAN.md) as the document lifecycle, while keeping the global `SNN` document sequence separate from phase numbers. Keep planning, environment bring-up, isolated diagnoses, unvalidated candidates, and other work below the threshold above in the existing phase reference; give each qualifying substantive stage deliverable one reproducible result document. Every retained optimization claim must name the actual source files and commits and summarize the correctness, performance, and rule result without citing internal experiment artifacts; hypotheses and official-platform verification gaps remain explicitly labeled inline.
