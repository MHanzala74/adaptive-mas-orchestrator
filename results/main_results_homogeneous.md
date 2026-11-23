## Mock environment: homogeneous

### unconstrained  (lam=0.3, 3 seeds x 200 tasks)

| method | accuracy [95% CI] | tokens/task | reward [95% CI] |
|---|---|---|---|
| single | 0.613 [0.567, 0.657] | 36 | 0.577 [0.530, 0.621] |
| debate | 0.827 [0.792, 0.860] | 255 | 0.572 [0.534, 0.607] |
| planner_worker | 0.670 [0.625, 0.713] | 80 | 0.590 [0.545, 0.635] |
| router (full-info) | 0.670 [0.625, 0.713] | 81 | 0.589 [0.543, 0.633] |
| router (bandit) | 0.823 [0.788, 0.858] | 254 | 0.570 [0.532, 0.606] |
| oracle | 0.938 [0.913, 0.960] | 69 | 0.870 [0.844, 0.893] |

router (full-info) - best fixed (planner_worker) reward: -0.002 [-0.004, 0.000] -> not significant

router (bandit) - best fixed (planner_worker) reward: -0.021 [-0.065, 0.021] -> not significant

### budget=150  (lam=0.3, 3 seeds x 200 tasks)

| method | accuracy [95% CI] | tokens/task | reward [95% CI] |
|---|---|---|---|
| single | 0.613 [0.567, 0.657] | 36 | 0.577 [0.530, 0.621] |
| debate | 0.733 [0.687, 0.778] | 168 | 0.565 [0.519, 0.610] |
| planner_worker | 0.670 [0.625, 0.713] | 80 | 0.590 [0.545, 0.635] |
| router (full-info) | 0.670 [0.625, 0.713] | 80 | 0.590 [0.545, 0.635] |
| router (bandit) | 0.670 [0.625, 0.713] | 80 | 0.590 [0.545, 0.635] |
| oracle | 0.840 [0.802, 0.875] | 46 | 0.794 [0.756, 0.829] |

router (full-info) - best fixed (planner_worker) reward: 0.000 [0.000, 0.000] -> not significant

router (bandit) - best fixed (planner_worker) reward: 0.000 [0.000, 0.000] -> not significant
