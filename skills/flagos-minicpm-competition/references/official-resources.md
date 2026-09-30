# Official resource map

Read this file before material architecture, kernel, scheduler, platform, or competition-rule work. Re-check live pages when the task depends on current rules or current software versions; this file is a routing map, not a frozen substitute for the sources.

## Authority order

1. [FlagOS Season 2 competition page](https://flagos.io/race-detail-season2?id=539vlt2p&lang=cn) and written organizer answers: repositories, revisions, commands, workloads, metrics, hardware, submission boundaries, and scoring rules.
2. [FlagOS documentation center](https://docs.flagos.io/zh-cn/latest/): FlagGems, FlagTree, vLLM-plugin-FL, FlagCX, KernelGen, release images, and multi-chip integration.
3. [FlagOS SkillHub](https://flagos.io/skillhub), [kernelgen-flagos page](https://flagos.io/SkillHubDetail?skillName=kernelgen-flagos), and the [flagos-ai/skills repository](https://github.com/flagos-ai/skills): official operator generation, optimization, specialization, MCP setup, testing, and feedback workflows.
4. Pinned source: [vllm-plugin-FL](https://github.com/flagos-ai/vllm-plugin-FL), [FlagGems](https://github.com/flagos-ai/FlagGems), and read-only [vLLM](https://github.com/vllm-project/vllm). Prefer the competition revisions over moving default branches.
5. Platform sources: the [Iluvatar developer community](https://developer.iluvatar.com/) and [Iluvatar corporate support entry](https://www.iluvatar.com/); the [MetaX developer documentation center](https://developer.metax-tech.com/doc). Use the exact SDK/driver/compiler version matching the official competition environment.
6. Upstream official specifications, documentation, source, issues, and pull requests for vLLM, Triton/FlagTree, PyTorch, and EvalScope. Use secondary material only when primary evidence is insufficient and label it clearly.

## Source-use checklist

For each material change, capture:

| Field | Required content |
|---|---|
| Question | The precise correctness, performance, platform, or rule question being answered |
| Source identity | URL, document title, repository, tag/commit, and access date |
| Applicable boundary | BI-V150/CoreX or C500/MACA, software version, dtype, shape, and execution phase |
| Adopted idea | The specific mechanism or invariant used by this project |
| Excluded idea | What was not copied because it violates rules, targets another platform, or changes semantics |
| Verification | Official-platform test/log/profile that can confirm the conclusion |

Never convert a current upstream default into a competition assumption. Capture the official environment’s resolved values from logs, including Attention backend, device identity, driver/runtime/compiler versions, imported package paths and commits, KV block size, token budget, chunked-prefill state, graph mode, and memory limit.

## Project-local evidence

- [`STATIC_ANALYSIS_AND_PLAN.md`](../../../STATIC_ANALYSIS_AND_PLAN.md) contains the initial source audit and hypotheses. Treat unmeasured performance conclusions as hypotheses.
- [`DETAILED_OPTIMIZATION_PLAN.md`](../../../DETAILED_OPTIMIZATION_PLAN.md) is the active seven-stage execution and acceptance plan.
- `vllm-plugin-FL` and `FlagGems` are the primary editable worktrees. `vllm` is an upstream read-only reference unless the organizer explicitly changes the submission scope.

When official sources disagree, the current competition page and direct written organizer answer win for contest behavior. Record the conflict and do not silently choose the more permissive interpretation.
