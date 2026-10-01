| Step | Records before | Records after | Dropped |
|---|---|---|---|
| raw survey records | 44,871 | 44,871 | 0 |
| valid NY coordinates | 44,871 | 44,871 | 0 |
| has a numeric Aqua_Pass_Score | 44,871 | 44,871 | 0 |
| score in [0, 1] (drops the -1 'no score - missing data' sentinel) | 44,871 | 37,808 | 7,063 |
| non-tidal (Tidal_Site != Yes) | 37,808 | 37,528 | 280 |
| accessible crossing with a structure | 37,528 | 34,729 | 2,799 |
| NAACC-approved records | 34,729 | 33,081 | 1,648 |
| collapse multi-structure rows (keep limiting structure) | 33,081 | 29,418 | 3,663 |
| keep most recent survey per crossing | 29,418 | 28,141 | 1,277 |

- Multi-structure rows are collapsed to the lowest-passability structure per (crossing code, survey date).
- Final modelling dataset: 28,141 unique surveyed crossings, 55 HUC8 watersheds, survey years 2001-2026.
- Target (passability) mean 0.614, median 0.695, fraction at 0.0: 4.8%, fraction at 1.0: 5.5%.