# 07 - Quick-check tier, volunteer-error sensitivity, and bias audit (2026-10-11)

Scripts: `scripts/plunkett_quickcheck_tier.py`, `scripts/plunkett_quickcheck_robust.py`. NY only (survey fields exist only for NY crossings
in our data; Plunkett's file has no field columns). Development = NY crossings outside the locked basins (14,061), HUC8 grouped CV, 3 splits.

## 1. What one cheap observation is worth (development set)

| Inputs | R2 |
|---|---|
| GIS only (T3 features) | 0.342 |
| + "outlet perched? yes/no" (drop > 0.25 ft; 28% yes) | 0.716 |
| + drop none / small / large | 0.766 |
| + perched + crossing type | 0.730 |
| + perched + type + constriction + inlet grade | 0.768 |

**Locked NY basins, scored once (2,094 crossings):** GIS only 0.287 -> GIS + perched yes/no **0.711**; MAE 0.212 -> 0.132.

## 2. Sensitivity to volunteer error (assumption: answers flipped at random with probability e, in train and test)

e = 0%: 0.716, 10%: 0.556, 20%: 0.467, 30%: 0.406. The gain is large but degrades quickly. The real volunteer error rate is **unmeasured**;
the symmetric random-flip model is an assumption. This is a different product (a rapid check that needs one field answer), NOT a GIS-only
prediction, and it must never be reported as the GIS-only accuracy. Caveat: the score is a function of the outlet drop, so part of the gain is
mechanical; the point is how much a seconds-long observation is worth, not that GIS alone reaches 0.7. A full NAACC survey already records outlet drop.

## 3. Bias audit of the GIS-only T3 model on the locked set (8,995)

Calibration by predicted-score quintile is close (predicted 0.41/0.55/0.64/0.73/0.92 vs observed 0.39/0.55/0.63/0.72/0.91).
Mean error (pred - obs) by state: MD -0.01, NY +0.01, PA +0.01, VA +0.01 (n >= 1,244); NJ +0.07 (n=128), DE -0.05 (n=40), WV +0.05 (n=31) - the last
three are too small to read. R2 by state: PA 0.33, VA 0.31, NY 0.28, MD 0.24.
**By structure type there is a real bias:** bridges are under-predicted (pred 0.84 vs obs 0.93; R2 within bridges -0.85, i.e. no ranking skill within
bridges), culverts over-predicted (0.58 vs 0.54; within-culvert R2 0.13). Part of the bridge gap is state "Bridge Adequate" practice (report 05, B7).
