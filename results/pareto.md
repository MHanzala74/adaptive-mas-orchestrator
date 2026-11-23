# Accuracy-cost trade-offs (simulated solver)

Hard-budget sweep: accuracy when each method (or the router, using its cost estimates) must stay within B tokens; over-budget runs return the best answer so far.

## homogeneous, independent errors

| method | B=60 | B=100 | B=150 | B=200 | B=300 | B=500 |
|---|---|---|---|---|---|---|
| single | 0.613 | 0.613 | 0.613 | 0.613 | 0.613 | 0.613 |
| adaptive_sc | 0.613 | 0.648 | 0.783 | 0.832 | 0.878 | 0.878 |
| sc7 | 0.613 | 0.648 | 0.782 | 0.837 | 0.908 | 0.908 |
| debate | 0.613 | 0.648 | 0.733 | 0.820 | 0.827 | 0.827 |
| debate_judge | 0.613 | 0.648 | 0.733 | 0.820 | 0.893 | 0.873 |
| router (core) | 0.613 | 0.670 | 0.780 | 0.780 | 0.780 | 0.780 |
| router (all) | 0.613 | 0.670 | 0.783 | 0.832 | 0.878 | 0.878 |

## heterogeneous, independent errors

| method | B=60 | B=100 | B=150 | B=200 | B=300 | B=500 |
|---|---|---|---|---|---|---|
| single | 0.557 | 0.557 | 0.557 | 0.557 | 0.557 | 0.557 |
| adaptive_sc | 0.557 | 0.592 | 0.712 | 0.755 | 0.815 | 0.815 |
| sc7 | 0.557 | 0.592 | 0.710 | 0.760 | 0.837 | 0.837 |
| debate | 0.557 | 0.592 | 0.670 | 0.765 | 0.858 | 0.858 |
| debate_judge | 0.557 | 0.592 | 0.670 | 0.765 | 0.905 | 0.890 |
| router (core) | 0.557 | 0.632 | 0.730 | 0.730 | 0.883 | 0.868 |
| router (all) | 0.557 | 0.632 | 0.712 | 0.755 | 0.823 | 0.870 |

## heterogeneous, correlated errors (0.5)

| method | B=60 | B=100 | B=150 | B=200 | B=300 | B=500 |
|---|---|---|---|---|---|---|
| single | 0.535 | 0.535 | 0.535 | 0.535 | 0.535 | 0.535 |
| adaptive_sc | 0.535 | 0.565 | 0.633 | 0.650 | 0.653 | 0.653 |
| sc7 | 0.535 | 0.565 | 0.633 | 0.675 | 0.707 | 0.707 |
| debate | 0.535 | 0.565 | 0.612 | 0.678 | 0.808 | 0.808 |
| debate_judge | 0.535 | 0.565 | 0.612 | 0.678 | 0.823 | 0.832 |
| router (core) | 0.535 | 0.610 | 0.668 | 0.668 | 0.820 | 0.830 |
| router (all) | 0.535 | 0.610 | 0.653 | 0.663 | 0.807 | 0.817 |

