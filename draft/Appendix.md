# Appendix

## 6.1

SVAR：人工损失

  | 扰动 | 最优 RL（历史反事实） | RL | History | Adv vs History | Taylor | Adv vs Taylor | 最优 RL（长期随机） | RL |
  Taylor | Adv vs Taylor |
  |---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
  | beta_097 | ppo_svar_direct_nonlinear | 57.87 | 55.81 | -3.69% | 60.16 | 3.80% | sac_svar_direct | 46.12 | 52.35 |
  11.91% |
  | beta_099 | ppo_ann_direct_nonlinear | 87.34 | 87.92 | 0.66% | 94.50 | 7.58% | sac_svar_direct | 96.59 | 112.06 |
  13.81% |
  | beta_0995 | ppo_ann_direct_nonlinear | 98.19 | 101.61 | 3.37% | 108.79 | 9.74% | sac_svar_direct | 123.94 | 144.20 |
  14.05% |
  | output_high | ppo_ann_direct_nonlinear | 123.69 | 130.55 | 5.26% | 142.92 | 13.46% | sac_svar_direct | 152.29 |
  173.75 | 12.35% |
  | output_low | ppo_ann_direct_nonlinear | 69.16 | 66.60 | -3.85% | 70.29 | 1.61% | sac_svar_direct | 68.74 | 81.22 |
  15.37% |
  | smooth_high | ppo_ann_direct_nonlinear | 87.98 | 89.19 | 1.36% | 95.12 | 7.51% | sac_svar_direct | 99.48 | 112.69 |
  11.73% |
  | smooth_low | ppo_ann_direct_nonlinear | 87.02 | 87.28 | 0.30% | 94.19 | 7.62% | sac_svar_direct | 95.14 | 111.74 |
  14.85% |



SVAR：反推损失

  | 扰动 | 最优 RL（历史反事实） | RL | History | Adv vs History | Taylor | Adv vs Taylor | 最优 RL（长期随机） | RL |
  Taylor | Adv vs Taylor |
  |---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
  | beta_097 | ppo_svar_direct | 80.73 | 228.71 | 64.70% | 138.85 | 41.86% | sac_svar_revealed_direct | 92.46 | 133.14 |
  30.56% |
  | beta_099 | ppo_svar_direct | 134.66 | 375.37 | 64.13% | 254.74 | 47.14% | sac_svar_revealed_direct | 202.92 | 285.87
  | 29.02% |
  | beta_0995 | ppo_svar_direct | 157.08 | 437.65 | 64.11% | 305.68 | 48.61% | sac_svar_revealed_direct | 263.57 |
  368.70 | 28.51% |
  | output_down | ppo_svar_direct | 105.15 | 337.78 | 68.87% | 212.05 | 50.41% | sac_svar_revealed_direct | 159.30 |
  231.48 | 31.18% |
  | output_up | ppo_svar_direct | 164.18 | 412.96 | 60.24% | 297.43 | 44.80% | sac_svar_revealed_direct | 246.54 |
  340.26 | 27.54% |
  | smooth_down | ppo_svar_direct | 130.59 | 247.28 | 47.19% | 192.79 | 32.27% | sac_svar_revealed_direct | 192.90 |
  222.19 | 13.18% |
  | smooth_up | ppo_svar_direct | 138.74 | 503.46 | 72.44% | 316.68 | 56.19% | sac_svar_revealed_direct | 212.94 |
  349.54 | 39.08% |



ANN：人工损失

  | 扰动 | 最优 RL（历史反事实） | RL | History | Adv vs History | Taylor | Adv vs Taylor | 最优 RL（长期随机） | RL |
  Taylor | Adv vs Taylor |
  |---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
  | beta_097 | ppo_ann_direct | 44.45 | 89.35 | 50.26% | 80.38 | 44.71% | ppo_ann_direct | 36.08 | 70.78 | 49.03% |
  | beta_099 | ppo_ann_direct | 69.23 | 168.24 | 58.85% | 130.41 | 46.91% | ppo_ann_direct | 75.12 | 162.73 | 53.84% |
  | beta_0995 | ppo_ann_direct | 79.97 | 205.07 | 61.01% | 151.28 | 47.14% | ppo_ann_direct | 96.12 | 212.95 | 54.86% |
  | output_high | ppo_ann_direct | 101.13 | 245.98 | 58.89% | 172.12 | 41.24% | ppo_ann_direct | 113.37 | 208.23 |
  45.55% |
  | output_low | td3_ann_direct | 49.35 | 129.37 | 61.85% | 109.55 | 54.95% | td3_ann_direct | 54.38 | 139.98 | 61.15% |
  | smooth_high | ppo_ann_direct | 70.64 | 169.52 | 58.33% | 131.02 | 46.08% | ppo_ann_direct | 77.07 | 163.21 | 52.78%
  |
  | smooth_low | td3_ann_direct | 65.96 | 167.60 | 60.65% | 130.10 | 49.30% | td3_ann_direct | 72.60 | 162.49 | 55.32% |



ANN：反推损失

  | 扰动 | 最优 RL（历史反事实） | RL | History | Adv vs History | Taylor | Adv vs Taylor | 最优 RL（长期随机） | RL |
  Taylor | Adv vs Taylor |
  |---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
  | beta_097 | td3_ann_revealed_direct | 116.93 | 272.78 | 57.13% | 168.06 | 30.42% | sac_ann_revealed_direct | 121.50 |
  137.46 | 11.61% |
  | beta_099 | td3_ann_revealed_direct | 186.23 | 483.77 | 61.50% | 285.35 | 34.74% | sac_ann_revealed_direct | 253.63 |
  292.72 | 13.35% |
  | beta_0995 | td3_ann_revealed_direct | 215.87 | 579.63 | 62.76% | 335.36 | 35.63% | sac_ann_revealed_direct | 324.55
  | 376.62 | 13.83% |
  | output_down | td3_ann_revealed_direct | 160.59 | 415.23 | 61.33% | 248.57 | 35.40% | sac_ann_revealed_direct |
  208.91 | 252.60 | 17.30% |
  | output_up | td3_ann_revealed_direct | 211.88 | 552.31 | 61.64% | 322.13 | 34.23% | sac_ann_revealed_direct | 298.35
  | 332.84 | 10.36% |
  | smooth_down | td3_ann_revealed_direct | 161.17 | 355.04 | 54.60% | 223.50 | 27.89% | sac_ann_revealed_direct |
  241.75 | 244.85 | 1.27% |
  | smooth_up | td3_ann_revealed_direct | 211.29 | 612.50 | 65.50% | 347.21 | 39.15% | sac_ann_revealed_direct | 265.52
  | 340.58 | 22.04% |
