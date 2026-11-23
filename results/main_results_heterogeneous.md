## Mock environment: heterogeneous

### unconstrained  (lam=0.3, 3 seeds x 200 tasks)

| method | accuracy [95% CI] | tokens/task | reward [95% CI] |
|---|---|---|---|
| single | 0.557 [0.512, 0.602] | 36 | 0.521 [0.475, 0.566] |
| debate | 0.858 [0.828, 0.887] | 255 | 0.603 [0.572, 0.633] |
| planner_worker | 0.632 [0.583, 0.677] | 80 | 0.552 [0.503, 0.598] |
| router (full-info) | 0.830 [0.798, 0.860] | 187 | 0.643 [0.605, 0.678] |
| router (bandit) | 0.858 [0.828, 0.887] | 255 | 0.603 [0.572, 0.633] |
| oracle | 0.945 [0.925, 0.963] | 82 | 0.863 [0.839, 0.884] |

router (full-info) - best fixed (debate) reward: 0.040 [0.015, 0.065] -> significant

router (bandit) - best fixed (debate) reward: 0.000 [0.000, 0.000] -> not significant

### budget=150  (lam=0.3, 3 seeds x 200 tasks)

| method | accuracy [95% CI] | tokens/task | reward [95% CI] |
|---|---|---|---|
| single | 0.557 [0.512, 0.602] | 36 | 0.521 [0.475, 0.566] |
| debate | 0.670 [0.623, 0.720] | 168 | 0.502 [0.454, 0.551] |
| planner_worker | 0.632 [0.583, 0.677] | 80 | 0.552 [0.503, 0.598] |
| router (full-info) | 0.632 [0.583, 0.677] | 80 | 0.552 [0.503, 0.598] |
| router (bandit) | 0.557 [0.512, 0.602] | 36 | 0.521 [0.475, 0.566] |
| oracle | 0.788 [0.747, 0.830] | 46 | 0.742 [0.700, 0.783] |

router (full-info) - best fixed (planner_worker) reward: 0.000 [0.000, 0.000] -> not significant

router (bandit) - best fixed (planner_worker) reward: -0.032 [-0.081, 0.021] -> not significant
