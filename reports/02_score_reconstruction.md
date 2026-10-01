# Phase 2 - reconstructing the NAACC passability score

**Goal:** confirm we understand exactly what the target variable
(`Aqua_Pass_Score`) represents, by recomputing it from the field-survey columns
using the *documented* NAACC 2016 algorithm and checking it against the value
NAACC publishes.

**Algorithm source:** "Aquatic Connectivity Scoring Systems for Non-tidal
Crossings" (rev. 2016-06-16), streamcontinuity.org. Implemented in
`src/naacc_score.py`. Also the algorithm the DSL / Plunkett model used as its
training target.

## Method

Per crossing:

1. **Continuous component scores** from smooth functions of the field
   measurements (openness ratio, structure height, outlet drop to water
   surface - all in feet):
   * `So = (1 - e^(-15 x (1-0.62)))^(1/(1-0.62))`
   * `Sh = min(1.1 x^2 / (2.2^2 + x^2), 1)`
   * `Sod = 1 - 1.029412 x^2 / (0.51449575^2 + x^2)`
2. **Categorical component scores** from lookup tables (constriction, inlet
   grade, physical-barrier severity, scour pool, substrate match / coverage,
   water depth / velocity, armoring, internal structures).
3. **Composite** = weighted sum (13 weights summing to 1.000; outlet drop 0.161
   is the largest). Components with no data have their weight redistributed
   across the components that do.
4. **Final score** = `min(composite, Sod)` - a large outlet drop caps the score.

## Result (28,141 unique surveyed crossings)

| metric | value |
|---|---|
| Pearson r (reconstruction vs. published) | **0.888** |
| Spearman rho | 0.882 |
| MAE | **0.055** |
| RMSE | 0.148 |
| bias (recon - published) | -0.014 |
| within 0.05 | 71.4% |
| within 0.10 | **92.2%** |
| within 0.20 | 96.2% |

The reconstruction is a faithful **approximation**, not a bit-exact
reproduction. That is expected: NAACC's production application has rules that
the public formula document does not fully specify.

## Where the reconstruction disagrees

The ~4% of crossings with error > 0.20 are concentrated in **cascading /
free-fall outlets**:

| Outlet grade | crossings with error > 0.25 |
|---|---|
| Free Fall | 279 |
| Free Fall Onto Cascade | 156 |
| Cascade | 68 |
| At Stream Grade | 49 |
| other | 12 |

For these, NAACC's application evidently applies a hard "No AOP" override
(consistent with the coarse-screen rule that a cascading outlet = No AOP) that
the documented weighted formula alone does not capture. Adding a naive cascade
cap made the overall fit *worse* (it over-penalised the free-fall crossings that
NAACC actually scores as moderate), so the reconstruction keeps the documented
formula and flags this as a known gap rather than guessing at the override.

Mean absolute error by AOP class:

| AOP class | mean abs error | n |
|---|---|---|
| No AOP | 0.034 | 10,003 |
| Reduced AOP | 0.054 | 12,517 |
| Full AOP | 0.102 | 5,000 |

The reconstruction is slightly conservative for Full-AOP crossings.

## Takeaway for the modelling project

The target `Aqua_Pass_Score` is a **deterministic function of the field-survey
measurements** (r = 0.89 from the public formula alone; the rest is NAACC's
undocumented overrides). It is *not* a subjective label. The GIS-prediction task
is therefore well-posed: learn the mapping *GIS predictors -> this score* for
crossings that have never been surveyed.
