# Impact measurement plan - NAACC passability project

Same discipline as the other projects here: separate demonstrated / prototype /
pilot / adopted / operational, and never claim the later stages before they
happen.

## Current status: demonstrated / prototype capability only

Nothing here has been shown to, evaluated by, or adopted by NAACC, NACC, or any
state DOT / conservation group. No crossing's real-world priority has been
changed by this work.

## What is actually demonstrated

| Metric | Value |
|---|---|
| Surveys in the multi-state file | 56,525 (13 states) |
| Locked test set | 8,995 crossings in 7 basins, scored once |
| Locked R2: random forest / full model | 0.286 / 0.320 |
| Difference | +0.034 (95% interval 0.026 to 0.045) |
| Predictions produced for unsurveyed crossings | none |
| Rankings changed in any organisation's list | none |

## What would move this forward (in order)

1. Plunkett's pipeline adds the NBI join (`docs/NBI_JOIN.md`) and reports whether his own held-out accuracy improves.
2. A field test of a rapid perched-outlet check, which would also measure volunteer error.
3. Re-link the stream attributes (StreamCat/NHDPlus) to the correct segments for all crossings.
4. Only after that, compare predicted rankings with an existing prioritisation list.

## Language to use later (not yet)

* "Model evaluated for integration" - only once someone at NAACC/NACC has
  actually looked at it.
* "Rankings changed" - only with a real before/after comparison against an
  existing operational prioritisation list, not this project's own catchment-
  only baseline.
* "Adopted" / "operational" - only if a program actually runs it repeatedly on
  their own workflow.
