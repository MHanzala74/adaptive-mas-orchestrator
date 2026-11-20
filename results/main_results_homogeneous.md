## Mock environment: homogeneous

### unconstrained  (lam=0.3, 3 seeds x 200 tasks)

| method | accuracy [95% CI] | tokens/task | reward [95% CI] |
|---|---|---|---|
| single | 0.613 [0.573, 0.653] | 36 | 0.577 [0.537, 0.617] |
| debate | 0.827 [0.795, 0.857] | 255 | 0.572 [0.539, 0.602] |
| planner_worker | 0.670 [0.635, 0.708] | 80 | 0.590 [0.555, 0.629] |
| router (full-info) | 0.670 [0.635, 0.708] | 81 | 0.589 [0.553, 0.627] |
| router (bandit) | 0.823 [0.792, 0.852] | 254 | 0.570 [0.537, 0.599] |
| oracle | 0.938 [0.918, 0.957] | 69 | 0.870 [0.850, 0.889] |

router (full-info) - best fixed (planner_worker) reward: -0.002 [-0.003, -0.001] -> significant

router (bandit) - best fixed (planner_worker) reward: -0.021 [-0.064, 0.021] -> not significant

### budget=150  (lam=0.3, 3 seeds x 200 tasks)

| method | accuracy [95% CI] | tokens/task | reward [95% CI] |
|---|---|---|---|
| single | 0.613 [0.573, 0.653] | 36 | 0.577 [0.537, 0.617] |
| debate | 0.733 [0.698, 0.768] | 168 | 0.565 [0.530, 0.600] |
| planner_worker | 0.670 [0.635, 0.708] | 80 | 0.590 [0.555, 0.629] |
| router (full-info) | 0.670 [0.635, 0.708] | 80 | 0.590 [0.555, 0.629] |
| router (bandit) | 0.670 [0.635, 0.708] | 80 | 0.590 [0.555, 0.629] |
| oracle | 0.840 [0.810, 0.868] | 46 | 0.794 [0.764, 0.823] |

router (full-info) - best fixed (planner_worker) reward: 0.000 [0.000, 0.000] -> not significant

router (bandit) - best fixed (planner_worker) reward: 0.000 [0.000, 0.000] -> not significant
