## Preliminary Seed-42 Results

> **Status:** single-seed results (`seed=42`). Multi-seed replication is in progress, so the numbers below should be treated as preliminary rather than as a final statistical claim.

### Endpoint comparison

![Seed-42 endpoint comparison](assets/results/seed42/seed42_endpoint_scores.png)

| Method | Step | Prompt pool | Generated rollouts / step | Total generated rollouts | Train rollouts / step | Validation score |
|---|---:|---:|---:|---:|---:|---:|
| Fixed-G5 @160 | 160 | 4 | 20 | 3200 | 20 | 79.76% |
| TARO-v1 @160 | 160 | 4 | 20 | 3200 | variable (~18) | 79.37% |
| TARO-v2 Base-G4 @160 | 160 | 4 | 20 | 3200 | 20 | 82.32% |
| Fixed-G8 @200 | 200 | 2 | 16 | 3200 | 16 | 81.93% |
| Fixed-G5 @200 | 200 | 4 | 20 | 4000 | 20 | 80.35% |
| TARO-v1 @200 | 200 | 4 | 20 | 4000 | variable (~18) | 81.34% |
| TARO-v2 Base-G4 @200 | 200 | 4 | 20 | 4000 | 20 | 82.71% |

### Controlled comparison: TARO-v2 Base-G4 vs Fixed-G5

![TARO-v2 vs Fixed-G5](assets/results/seed42/seed42_g5_vs_v2.png)

The cleanest controlled comparison keeps the prompt pool, rollout budget, training-step count, and number of train rollouts matched:

- **@200:** TARO-v2 Base-G4 improves over Fixed-G5 by **+2.36 pp**, paired bootstrap 95% CI **[+0.00, +4.91] pp**, exact McNemar `p=0.088`.
- **@160:** TARO-v2 Base-G4 improves over Fixed-G5 by **+2.55 pp**, paired bootstrap 95% CI **[+0.00, +5.11] pp**, exact McNemar `p=0.079`.

### Structural ablation

![Seed-42 pairwise effects](assets/results/seed42/seed42_pairwise_effects.png)

Removing the early hard prompt-admission stage and using mandatory Base-G4 yields:

- **v2@200 − v1@200:** **+1.38 pp**, 95% CI **[-1.38, +4.13] pp**.
- **v2@160 − v1@160:** **+2.95 pp**, 95% CI **[+0.39, +5.70] pp**.

All endpoint evaluations use the same deterministic validation setup with paired `N=509`.
