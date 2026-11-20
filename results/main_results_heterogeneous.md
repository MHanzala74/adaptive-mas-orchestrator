## Mock environment: heterogeneous

### unconstrained  (lam=0.3, 3 seeds x 200 tasks)

| method | accuracy [95% CI] | tokens/task | reward [95% CI] |
|---|---|---|---|
| single | 0.557 [0.517, 0.597] | 36 | 0.521 [0.480, 0.561] |
| debate | 0.858 [0.830, 0.885] | 255 | 0.603 [0.574, 0.631] |
| planner_worker | 0.632 [0.595, 0.672] | 80 | 0.552 [0.515, 0.592] |
| router (full-info) | 0.830 [0.800, 0.858] | 187 | 0.643 [0.612, 0.675] |
| router (bandit) | 0.858 [0.830, 0.885] | 255 | 0.603 [0.574, 0.631] |
| oracle | 0.945 [0.925, 0.963] | 82 | 0.863 [0.843, 0.881] |

router (full-info) - best fixed (debate) reward: 0.040 [0.018, 0.062] -> significant

router (bandit) - best fixed (debate) reward: 0.000 [0.000, 0.000] -> not significant

### budget=150  (lam=0.3, 3 seeds x 200 tasks)

| method | accuracy [95% CI] | tokens/task | reward [95% CI] |
|---|---|---|---|
| single | 0.557 [0.517, 0.597] | 36 | 0.521 [0.480, 0.561] |
| debate | 0.670 [0.633, 0.708] | 168 | 0.502 [0.464, 0.540] |
| planner_worker | 0.632 [0.595, 0.672] | 80 | 0.552 [0.515, 0.592] |
| router (full-info) | 0.632 [0.595, 0.672] | 80 | 0.552 [0.515, 0.592] |
| router (bandit) | 0.557 [0.517, 0.597] | 36 | 0.521 [0.480, 0.561] |
| oracle | 0.788 [0.755, 0.822] | 46 | 0.742 [0.710, 0.775] |

router (full-info) - best fixed (planner_worker) reward: 0.000 [0.000, 0.000] -> not significant

router (bandit) - best fixed (planner_worker) reward: -0.032 [-0.083, 0.017] -> not significant
