# Baselines vs orchestrated architectures (simulated solver)

3 seeds x 200 test tasks; reward = correct - 0.3 * tokens/300; CIs: 95% cluster bootstrap over tasks. `router (core)` chooses among the 5 core architectures; `router (all)` also among the baselines.

## Environment: homogeneous, independent errors

| method | accuracy [95% CI] | tokens/task | reward [95% CI] |
|---|---|---|---|
| single | 0.613 [0.567, 0.657] | 36 | 0.577 [0.530, 0.621] |
| planner_worker | 0.670 [0.625, 0.713] | 80 | 0.590 [0.545, 0.635] |
| solver_critic | 0.780 [0.740, 0.815] | 108 | 0.672 [0.631, 0.710] |
| router (core) | 0.780 [0.740, 0.815] | 108 | 0.672 [0.631, 0.710] |
| sc3 | 0.727 [0.680, 0.772] | 108 | 0.619 [0.572, 0.665] |
| adaptive_sc | 0.878 [0.847, 0.908] | 120 | 0.758 [0.721, 0.793] |
| router (all) | 0.878 [0.847, 0.908] | 120 | 0.758 [0.721, 0.793] |
| sc5 | 0.837 [0.800, 0.870] | 180 | 0.657 [0.618, 0.692] |
| cascade | 0.857 [0.823, 0.887] | 225 | 0.632 [0.589, 0.670] |
| sc7 | 0.908 [0.880, 0.933] | 252 | 0.656 [0.626, 0.684] |
| debate | 0.827 [0.792, 0.860] | 255 | 0.572 [0.534, 0.607] |
| debate_judge | 0.873 [0.842, 0.903] | 263 | 0.611 [0.576, 0.643] |

Paired differences (best fixed by reward = `adaptive_sc`):
- accuracy: `debate` - `sc7` = -0.082 [-0.113, -0.050] (significant)
- accuracy: `debate_judge` - `sc7` = -0.035 [-0.062, -0.008] (significant)
- accuracy: `adaptive_sc` - `debate` = 0.052 [0.017, 0.085] (significant)
- reward: `router (core)` - `adaptive_sc` = -0.085 [-0.117, -0.056] (significant)
- reward: `router (all)` - `adaptive_sc` = 0.000 [0.000, 0.000] (n.s.)

## Environment: heterogeneous, independent errors

| method | accuracy [95% CI] | tokens/task | reward [95% CI] |
|---|---|---|---|
| single | 0.557 [0.512, 0.602] | 36 | 0.521 [0.475, 0.566] |
| planner_worker | 0.632 [0.583, 0.677] | 80 | 0.552 [0.503, 0.598] |
| sc3 | 0.663 [0.615, 0.712] | 108 | 0.555 [0.506, 0.605] |
| solver_critic | 0.733 [0.688, 0.773] | 110 | 0.624 [0.577, 0.666] |
| adaptive_sc | 0.815 [0.775, 0.852] | 131 | 0.684 [0.639, 0.727] |
| router (all) | 0.870 [0.840, 0.898] | 163 | 0.707 [0.667, 0.745] |
| sc5 | 0.760 [0.715, 0.803] | 180 | 0.580 [0.533, 0.626] |
| router (core) | 0.868 [0.837, 0.897] | 187 | 0.681 [0.644, 0.718] |
| cascade | 0.873 [0.847, 0.900] | 236 | 0.638 [0.602, 0.671] |
| sc7 | 0.837 [0.800, 0.872] | 252 | 0.585 [0.546, 0.621] |
| debate | 0.858 [0.828, 0.887] | 255 | 0.603 [0.572, 0.633] |
| debate_judge | 0.890 [0.862, 0.917] | 262 | 0.628 [0.597, 0.657] |

Paired differences (best fixed by reward = `adaptive_sc`):
- accuracy: `debate` - `sc7` = 0.022 [-0.015, 0.057] (n.s.)
- accuracy: `debate_judge` - `sc7` = 0.053 [0.022, 0.085] (significant)
- accuracy: `adaptive_sc` - `debate` = -0.043 [-0.080, -0.007] (significant)
- reward: `router (core)` - `adaptive_sc` = -0.002 [-0.039, 0.034] (n.s.)
- reward: `router (all)` - `adaptive_sc` = 0.023 [0.005, 0.043] (significant)

## Environment: heterogeneous, correlated errors (0.5)

| method | accuracy [95% CI] | tokens/task | reward [95% CI] |
|---|---|---|---|
| single | 0.535 [0.488, 0.580] | 36 | 0.499 [0.452, 0.545] |
| planner_worker | 0.610 [0.562, 0.655] | 80 | 0.530 [0.481, 0.576] |
| solver_critic | 0.675 [0.628, 0.720] | 108 | 0.567 [0.519, 0.614] |
| sc3 | 0.603 [0.553, 0.652] | 108 | 0.495 [0.444, 0.545] |
| adaptive_sc | 0.653 [0.605, 0.702] | 111 | 0.543 [0.491, 0.595] |
| sc5 | 0.673 [0.622, 0.722] | 180 | 0.493 [0.441, 0.544] |
| router (all) | 0.817 [0.783, 0.850] | 183 | 0.634 [0.591, 0.674] |
| router (core) | 0.830 [0.795, 0.865] | 222 | 0.608 [0.567, 0.648] |
| cascade | 0.790 [0.753, 0.827] | 225 | 0.565 [0.521, 0.605] |
| sc7 | 0.707 [0.658, 0.753] | 252 | 0.455 [0.405, 0.504] |
| debate | 0.808 [0.773, 0.843] | 255 | 0.553 [0.517, 0.589] |
| debate_judge | 0.832 [0.797, 0.865] | 260 | 0.572 [0.535, 0.607] |

Paired differences (best fixed by reward = `debate_judge`):
- accuracy: `debate` - `sc7` = 0.102 [0.063, 0.142] (significant)
- accuracy: `debate_judge` - `sc7` = 0.125 [0.087, 0.165] (significant)
- accuracy: `adaptive_sc` - `debate` = -0.155 [-0.198, -0.113] (significant)
- reward: `router (core)` - `debate_judge` = 0.036 [0.025, 0.048] (significant)
- reward: `router (all)` - `debate_judge` = 0.062 [0.042, 0.081] (significant)

