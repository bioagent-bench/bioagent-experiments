# BioAgent Bench — Key Takeaways

Synthesis of the original paper results and the updated model panel
(Figure 2 heatmap re-run with current-generation models; Figure 3 plan-quality
vs. completion).

## Headline findings

1. **Frontier agents complete canonical bioinformatics pipelines end-to-end.**
   Current top models cluster at the ceiling: Claude Opus 4.7 and GPT-5.5 both
   average **98%** task completion, Gemini 3 Pro **96.6%**, Claude Sonnet 4.5
   **92.5%**. State-of-the-art agents can act as effective workflow assistants
   for common analyses without bespoke scaffolding.

2. **Newer is not uniformly better — version upgrades can regress.**
   Holding the harness fixed (opencode), within-family upgrades move in *both*
   directions:
   - Claude Opus 4.5 → 4.7: 93.3 → **98.0** (+4.7)
   - GPT-5.2 → GPT-5.5: 87.5 → **98.0** (+10.5)
   - Qwen3 Coder → Qwen3.7 Max: 68.6 → **83.0** (+14.4)
   - **Kimi K2 Thinking → K2.6: 80.5 → 58.0 (−22.5, a sharp regression)**

   Benchmarking must be re-run per release; a newer checkpoint is not a safe
   drop-in assumption.

3. **The open/closed gap is narrowing but persists.** The strongest open-weight
   model (Qwen3.7 Max, 83%) now rivals mid-tier closed models, yet open-weight
   results span a wide band down to 58% (Kimi K2.6) and remain more variable.
   Open-weight models matter most where data privacy forbids routing to closed
   providers (clinical, proprietary, or unpublished sequencing data).

4. **The agent harness is a first-class experimental variable.** The *same*
   underlying model swings dramatically with the scaffold:
   - Gemini 3 Pro: **96.6%** (Codex CLI) vs **68.0%** (opencode)
   - Claude Opus 4.5: **100%** / 96% / 93% across Codex CLI / Claude Code / opencode
   - GPT-5.1 family: 38.5% → 74.7% → 81.7% across plain / codex / codex-max
   - Qwen-coder: **0%** (Codex CLI) vs 68.6% (opencode)

   Swings of 25–95 points mean any leaderboard is only meaningful with the
   harness reported alongside the model.

## Reliability caveats (completion ≠ trustworthiness)

5. **Pipeline completion overstates reliability.** Across repeated trials the
   final results are only moderately stable — mean Jaccard overlap **0.43**,
   mean Pearson correlation **0.73** — reflecting genuine analytical degrees of
   freedom plus agent non-determinism (tool flags, statistical choices).

6. **Agents are weak at *not* proceeding on bad inputs.** Under perturbation,
   models correctly flagged corrupted inputs in only **7/10** tasks and
   erroneously used decoy files in **2/10**. Prompt bloat dropped completion by
   ~**28%** on average, with several tasks collapsing entirely
   (deseq, metagenomics, single-cell: **−100%**). High-level pipeline assembly
   does not imply correct step-level reasoning.

7. **Plan quality predicts, but does not determine, success.** Explicit plan
   ratings correlate with completion (Pearson **r = 0.61**), so better planning
   generally helps — but the link is not deterministic: some models complete
   tasks despite weak plans, showing agentic execution can compensate for
   shallow explicit planning.

## Implication

For sensitive use (e.g., clinical diagnostics), the relevant question is **not
"can it produce a result?"** — frontier agents largely can — **but "can it
reliably detect when it should *not* proceed, and justify choices with evidence
grounded in the data?"** Treat pipeline completion as a necessary-but-insufficient
metric, and always report the harness used.

---
*Notes on data provenance:* Version-to-version deltas in point 2 use
opencode-harness numbers for an apples-to-apples comparison. The Figure 2 heatmap
mixes harnesses (Codex CLI for the kept frontier models, opencode for the newer
checkpoints), so absolute cross-family ordering there should be read with the
harness caveat in point 4. Robustness/perturbation figures (points 5–6) are from
the GPT-5.2 Codex CLI runs in the original paper.
