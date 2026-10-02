# Mean and Standard Deviation Calculations

 Scope: 18 model-level results across Clean LIBERO, LIBERO-Plus Robust Avg., and LIBERO-Para, plus the OpenPI Clean-to-Para drop.

The calculations below use the listed inputs and the population standard deviation (`ddof=0`). First compute the model-level score within each run, then calculate the mean and standard deviation across the run scores. Standard deviations are expressed in percentage points (pp).

## 1. Results reported in the paper

```text
+-------------------+------------------+-----------------+-----------------+
| Model             | Clean LIBERO (%) | LIBERO-Plus (%) | LIBERO-Para (%) |
+-------------------+------------------+-----------------+-----------------+
| OpenPI            |   96.52 +/- 0.22 |  85.31 +/- 0.32 |  71.33 +/- 0.05 |
| UnifoLM-VLA-0     |   97.92 +/- 0.06 |  78.78 +/- 0.30 |  82.24 +/- 0.48 |
| Xiaomi-Robotics-0 |   98.10 +/- 0.43 |  73.91 +/- 0.13 |  75.73 +/- 0.16 |
| GR00T-N1.7        |   97.88 +/- 0.65 |  75.12 +/- 0.28 |  74.26 +/- 1.52 |
| FastWAM           |   96.55 +/- 0.16 |  70.69 +/- 0.52 |  51.16 +/- 0.38 |
| Cosmos Policy     |   97.63 +/- 0.23 |  80.47 +/- 0.23 |  70.50 +/- 0.26 |
+-------------------+------------------+-----------------+-----------------+
```

## 2. Formulas and calculation order

R1, R2, and R3 denote the three run results used in the calculations below. The population standard deviation of these three inputs uses a denominator of 3.

```text
Input run scores: x1, x2, x3

mu     = (x1 + x2 + x3) / 3
var    = [(x1 - mu)^2 + (x2 - mu)^2 + (x3 - mu)^2] / 3
SD_pop = sqrt(var)
Result = mu +/- SD_pop
```

In these calculations, ± denotes the population standard deviation of the three run scores. It is distinct from a maximum deviation, range, standard error, or sample standard deviation. The sample standard deviation uses a denominator of 2 and equals `sqrt(3/2) * SD_pop` for three inputs.

Clean LIBERO: within each run, average the Spatial, Object, Goal, and Long suite scores.

```text
x_r = (Spatial_r + Object_r + Goal_r + Long_r) / 4
```

LIBERO-Plus: within each run, average the six non-linguistic perturbation dimensions. Language is excluded from Robust Avg. The Total field in the summaries is a different metric and does not replace this six-dimension average.

```text
x_r = (Camera_r + Robot_r + Light_r + Background_r + Noise_r + Layout_r) / 6
```

LIBERO-Para: divide each run's success count by 4,092 and multiply by 100, then calculate the mean and standard deviation across the three runs.

```text
x_r = 100 * successes_r / 4092
```

## 3. Clean LIBERO: run averages and standard deviations

The four inputs are ordered as Spatial, Object, Goal, and Long. Take their unweighted mean within each run, then calculate the mean and standard deviation across runs.

### OpenPI

Inputs: the three sets of Spatial, Object, Goal, and Long success rates shown below.

```text
x1 = (98.40 + 98.00 + 98.00 + 92.20) / 4 = 96.650000000
x2 = (98.40 + 99.20 + 97.80 + 91.40) / 4 = 96.700000000
x3 = (98.20 + 99.00 + 97.00 + 90.60) / 4 = 96.200000000

mu = (96.650000000 + 96.700000000 + 96.200000000) / 3
   = 96.516666667
SD = sqrt([(96.650000000 - 96.516666667)^2
         + (96.700000000 - 96.516666667)^2
         + (96.200000000 - 96.516666667)^2] / 3)
   = sqrt(0.050555555556)
   = 0.224845626 -> 0.22 pp
mu +/- SD = 96.52 +/- 0.22
```

### UnifoLM-VLA-0

Source: [evaluation records](results/unifolm-vla/results/LIBERO/three_seed_average.md).

```text
x1 = (99.40 + 100.00 + 97.80 + 94.80) / 4 = 98.000000000
x2 = (98.80 + 100.00 + 97.40 + 95.20) / 4 = 97.850000000
x3 = (97.80 + 100.00 + 98.00 + 95.80) / 4 = 97.900000000

mu = (98.000000000 + 97.850000000 + 97.900000000) / 3
   = 97.916666667
SD = sqrt([(98.000000000 - 97.916666667)^2
         + (97.850000000 - 97.916666667)^2
         + (97.900000000 - 97.916666667)^2] / 3)
   = sqrt(0.003888888889)
   = 0.062360956 -> 0.06 pp
mu +/- SD = 97.92 +/- 0.06
```

### Xiaomi-Robotics-0

Source: [evaluation records](results/Xiaomi-Robotics-0/results/LIBERO/three_seed_average.md).

```text
x1 = (98.80 + 100.00 + 96.60 + 94.60) / 4 = 97.500000000
x2 = (98.60 + 100.00 + 98.60 + 96.00) / 4 = 98.300000000
x3 = (98.60 + 100.00 + 98.60 + 96.80) / 4 = 98.500000000

mu = (97.500000000 + 98.300000000 + 98.500000000) / 3
   = 98.100000000
SD = sqrt([(97.500000000 - 98.100000000)^2
         + (98.300000000 - 98.100000000)^2
         + (98.500000000 - 98.100000000)^2] / 3)
   = sqrt(0.186666666667)
   = 0.432049380 -> 0.43 pp
mu +/- SD = 98.10 +/- 0.43
```

### FastWAM

Source: [evaluation records](results/FastWAM/result/three_seed_summary.json).

```text
x1 = (97.40 + 99.20 + 96.60 + 93.00) / 4 = 96.550000000
x2 = (96.80 + 99.20 + 95.80 + 93.60) / 4 = 96.350000000
x3 = (97.00 + 99.00 + 97.00 + 94.00) / 4 = 96.750000000

mu = (96.550000000 + 96.350000000 + 96.750000000) / 3
   = 96.550000000
SD = sqrt([(96.550000000 - 96.550000000)^2
         + (96.350000000 - 96.550000000)^2
         + (96.750000000 - 96.550000000)^2] / 3)
   = sqrt(0.026666666667)
   = 0.163299316 -> 0.16 pp
mu +/- SD = 96.55 +/- 0.16
```

### Cosmos Policy

Source: [evaluation records](results/cosmos-policy/results/libero_3seeds/libero_allgpu_run).

```text
x1 = (96.40 + 100.00 + 98.40 + 97.00) / 4 = 97.950000000
x2 = (96.00 + 99.60 + 97.60 + 96.40) / 4 = 97.400000000
x3 = (96.80 + 99.20 + 97.80 + 96.40) / 4 = 97.550000000

mu = (97.950000000 + 97.400000000 + 97.550000000) / 3
   = 97.633333333
SD = sqrt([(97.950000000 - 97.633333333)^2
         + (97.400000000 - 97.633333333)^2
         + (97.550000000 - 97.633333333)^2] / 3)
   = sqrt(0.053888888889)
   = 0.232139805 -> 0.23 pp
mu +/- SD = 97.63 +/- 0.23
```

## 4. LIBERO-Plus: six-dimension Robust Avg. and standard deviation

```text
+-------------------+-----------+-----------+-----------+-----------+----------+
| Model             |        R1 |        R2 |        R3 |  Mean (%) |  SD (pp) |
+-------------------+-----------+-----------+-----------+-----------+----------+
| OpenPI            | 85.006667 | 85.755000 | 85.160000 | 85.307222 | 0.322755 |
| UnifoLM-VLA-0     | 78.440651 | 79.159784 | 78.732390 | 78.777608 | 0.295321 |
| Xiaomi-Robotics-0 | 73.773459 | 73.857253 | 74.086813 | 73.905842 | 0.132460 |
| GR00T-N1.7        | 74.936667 | 74.898333 | 75.513333 | 75.116111 | 0.281314 |
| FastWAM           | 70.184064 | 70.466723 | 71.404323 | 70.685037 | 0.521538 |
| Cosmos Policy     | 80.793262 | 80.255982 | 80.375553 | 80.474932 | 0.230325 |
+-------------------+-----------+-----------+-----------+-----------+----------+
```

The dimensions are ordered as Camera, Robot, Light, Background, Noise, and Layout. OpenPI and GR00T use the two-decimal inputs in the run summaries; the other models use ratios calculated from success counts and denominators. The substituted values are displayed to nine decimal places.

### OpenPI

Source: [evaluation records](results/openpi/results/libero-plus/three_seed_average.md).

```text
x1 = (70.920000000 + 74.520000000 + 96.230000000 + 95.260000000 + 87.010000000 + 86.100000000) / 6 = 85.006666667
x2 = (70.860000000 + 76.190000000 + 97.370000000 + 95.910000000 + 87.510000000 + 86.690000000) / 6 = 85.755000000
x3 = (70.110000000 + 74.650000000 + 96.670000000 + 96.280000000 + 86.630000000 + 86.620000000) / 6 = 85.160000000

mu = (85.006666667 + 85.755000000 + 85.160000000) / 3
   = 85.307222222
SD = sqrt([(85.006666667 - 85.307222222)^2
         + (85.755000000 - 85.307222222)^2
         + (85.160000000 - 85.307222222)^2] / 3)
   = sqrt(0.104170987654)
   = 0.322755306 -> 0.32 pp
mu +/- SD = 85.31 +/- 0.32
```

### UnifoLM-VLA-0

Source: [evaluation records](results/unifolm-vla/results/LIBERO-plus).

```text
x1 = (56.722951845 + 67.225806452 + 93.520140105 + 95.167286245 + 79.450343535 + 78.557377049) / 6 = 78.440650872
x2 = (57.446808511 + 68.516129032 + 94.045534151 + 95.353159851 + 79.675810474 + 79.921259843) / 6 = 79.159783644
x3 = (56.910569106 + 69.354838710 + 93.520140105 + 94.888475836 + 78.638351031 + 79.081967213) / 6 = 78.732390333

mu = (78.440650872 + 79.159783644 + 78.732390333) / 3
   = 78.777608283
SD = sqrt([(78.440650872 - 78.777608283)^2
         + (79.159783644 - 78.777608283)^2
         + (78.732390333 - 78.777608283)^2] / 3)
   = sqrt(0.087214322016)
   = 0.295320710 -> 0.30 pp
mu +/- SD = 78.78 +/- 0.30
```

### Xiaomi-Robotics-0

Source: [evaluation records](results/Xiaomi-Robotics-0/results/LIBERO-plus).

```text
x1 = (40.087554722 + 55.741935484 + 93.957968476 + 90.613382900 + 86.633354154 + 75.606557377) / 6 = 73.773458852
x2 = (40.275171982 + 55.354838710 + 95.008756567 + 89.684014870 + 86.820737039 + 76.000000000) / 6 = 73.857253195
x3 = (40.275171982 + 55.806451613 + 94.570928196 + 90.985130112 + 86.883198001 + 76.000000000) / 6 = 74.086813317

mu = (73.773458852 + 73.857253195 + 74.086813317) / 3
   = 73.905841788
SD = sqrt([(73.773458852 - 73.905841788)^2
         + (73.857253195 - 73.905841788)^2
         + (74.086813317 - 73.905841788)^2] / 3)
   = sqrt(0.017545595856)
   = 0.132459790 -> 0.13 pp
mu +/- SD = 73.91 +/- 0.13
```

### GR00T-N1.7

Source: [evaluation records](results/Isaac-GR00T/results/libero-plus/three_seed_average.md).

```text
x1 = (64.380000000 + 39.580000000 + 93.550000000 + 93.650000000 + 84.130000000 + 74.330000000) / 6 = 74.936666667
x2 = (63.330000000 + 36.180000000 + 95.750000000 + 93.530000000 + 84.700000000 + 75.900000000) / 6 = 74.898333333
x3 = (65.500000000 + 39.450000000 + 95.300000000 + 93.980000000 + 84.250000000 + 74.600000000) / 6 = 75.513333333

mu = (74.936666667 + 74.898333333 + 75.513333333) / 3
   = 75.116111111
SD = sqrt([(74.936666667 - 75.116111111)^2
         + (74.898333333 - 75.116111111)^2
         + (75.513333333 - 75.116111111)^2] / 3)
   = sqrt(0.079137654321)
   = 0.281314156 -> 0.28 pp
mu +/- SD = 75.12 +/- 0.28
```

### FastWAM

Source: [evaluation records](results/FastWAM/results_plus).

```text
x1 = (44.402751720 + 71.677419355 + 92.732049037 + 65.799256506 + 67.083073079 + 79.409836066) / 6 = 70.184064294
x2 = (42.901813634 + 72.516129032 + 94.483362522 + 66.821561338 + 66.208619613 + 79.868852459) / 6 = 70.466723100
x3 = (46.654158849 + 71.935483871 + 95.446584939 + 67.100371747 + 68.207370394 + 79.081967213) / 6 = 71.404322835

mu = (70.184064294 + 70.466723100 + 71.404322835) / 3
   = 70.685036743
SD = sqrt([(70.184064294 - 70.685036743)^2
         + (70.466723100 - 70.685036743)^2
         + (71.404322835 - 70.685036743)^2] / 3)
   = sqrt(0.272002241568)
   = 0.521538341 -> 0.52 pp
mu +/- SD = 70.69 +/- 0.52
```

### Cosmos Policy

Source: [evaluation records](results/cosmos-policy/results/libero_plus_3seeds/cosmos_libero_plus_3seeds_20260719_082355).

```text
x1 = (73.233270794 + 52.645161290 + 98.949211909 + 85.223048327 + 90.381011868 + 84.327868852) / 6 = 80.793262173
x2 = (71.857410882 + 51.741935484 + 98.423817863 + 85.315985130 + 90.131168020 + 84.065573770) / 6 = 80.255981858
x3 = (72.858036273 + 53.290322581 + 98.073555166 + 84.293680297 + 90.131168020 + 83.606557377) / 6 = 80.375553286

mu = (80.793262173 + 80.255981858 + 80.375553286) / 3
   = 80.474932439
SD = sqrt([(80.793262173 - 80.474932439)^2
         + (80.255981858 - 80.474932439)^2
         + (80.375553286 - 80.474932439)^2] / 3)
   = sqrt(0.053049797582)
   = 0.230325417 -> 0.23 pp
mu +/- SD = 80.47 +/- 0.23
```

## 5. LIBERO-Para: success rates and standard deviations

```text
+-------------------+-----------+-----------+-----------+-----------+----------+
| Model             |        R1 |        R2 |        R3 |  Mean (%) |  SD (pp) |
+-------------------+-----------+-----------+-----------+-----------+----------+
| OpenPI            | 71.334311 | 71.383187 | 71.260997 | 71.326165 | 0.050215 |
| UnifoLM-VLA-0     | 81.891496 | 82.917889 | 81.915934 | 82.241773 | 0.478190 |
| Xiaomi-Robotics-0 | 75.708700 | 75.928641 | 75.537634 | 75.724992 | 0.160043 |
| GR00T-N1.7        | 75.537634 | 75.122190 | 72.116325 | 74.258716 | 1.524364 |
| FastWAM           | 51.612903 | 50.684262 | 51.173021 | 51.156729 | 0.379291 |
| Cosmos Policy     | 70.698925 | 70.136852 | 70.674487 | 70.503421 | 0.259395 |
+-------------------+-----------+-----------+-----------+-----------+----------+
```

Each run contains 4,092 episodes. Calculations use integer success counts directly, without first rounding the success rates.

```text
+-------------------+--------------+--------------+--------------+--------------+
| Model             | R1 successes | R2 successes | R3 successes | Episodes/run |
+-------------------+--------------+--------------+--------------+--------------+
| OpenPI            |         2919 |         2921 |         2916 |         4092 |
| UnifoLM-VLA-0     |         3351 |         3393 |         3352 |         4092 |
| Xiaomi-Robotics-0 |         3098 |         3107 |         3091 |         4092 |
| GR00T-N1.7        |         3091 |         3074 |         2951 |         4092 |
| FastWAM           |         2112 |         2074 |         2094 |         4092 |
| Cosmos Policy     |         2893 |         2870 |         2892 |         4092 |
+-------------------+--------------+--------------+--------------+--------------+
```

### OpenPI

Source: [evaluation records](results/openpi/results/libero_para/three_seed_average.md).

```text
x1 = 100 * 2919 / 4092 = 71.334310850
x2 = 100 * 2921 / 4092 = 71.383186706
x3 = 100 * 2916 / 4092 = 71.260997067

mu = (71.334310850 + 71.383186706 + 71.260997067) / 3
   = 71.326164875
SD = sqrt([(71.334310850 - 71.326164875)^2
         + (71.383186706 - 71.326164875)^2
         + (71.260997067 - 71.326164875)^2] / 3)
   = sqrt(0.002521563080)
   = 0.050215168 -> 0.05 pp
mu +/- SD = 71.33 +/- 0.05
```

### UnifoLM-VLA-0

Source: [evaluation records](results/unifolm-vla/results/libero_para/three_seed_average.md).

```text
x1 = 100 * 3351 / 4092 = 81.891495601
x2 = 100 * 3393 / 4092 = 82.917888563
x3 = 100 * 3352 / 4092 = 81.915933529

mu = (81.891495601 + 82.917888563 + 81.915933529) / 3
   = 82.241772564
SD = sqrt([(81.891495601 - 82.241772564)^2
         + (82.917888563 - 82.241772564)^2
         + (81.915933529 - 82.241772564)^2] / 3)
   = sqrt(0.228665957232)
   = 0.478190294 -> 0.48 pp
mu +/- SD = 82.24 +/- 0.48
```

### Xiaomi-Robotics-0

Source: [evaluation records](results/Xiaomi-Robotics-0/results/libero_para/three_seed_average.md).

```text
x1 = 100 * 3098 / 4092 = 75.708699902
x2 = 100 * 3107 / 4092 = 75.928641251
x3 = 100 * 3091 / 4092 = 75.537634409

mu = (75.708699902 + 75.928641251 + 75.537634409) / 3
   = 75.724991854
SD = sqrt([(75.708699902 - 75.724991854)^2
         + (75.928641251 - 75.724991854)^2
         + (75.537634409 - 75.724991854)^2] / 3)
   = sqrt(0.025613772342)
   = 0.160043033 -> 0.16 pp
mu +/- SD = 75.72 +/- 0.16
```

### GR00T-N1.7

Source: [evaluation records](results/Isaac-GR00T/results/libero_para/three_seed_average.md).

```text
x1 = 100 * 3091 / 4092 = 75.537634409
x2 = 100 * 3074 / 4092 = 75.122189638
x3 = 100 * 2951 / 4092 = 72.116324536

mu = (75.537634409 + 75.122189638 + 72.116324536) / 3
   = 74.258716194
SD = sqrt([(75.537634409 - 74.258716194)^2
         + (75.122189638 - 74.258716194)^2
         + (72.116324536 - 74.258716194)^2] / 3)
   = sqrt(2.323686735442)
   = 1.524364371 -> 1.52 pp
mu +/- SD = 74.26 +/- 1.52
```

### FastWAM

Source: [evaluation records](results/FastWAM/results_para).

```text
x1 = 100 * 2112 / 4092 = 51.612903226
x2 = 100 * 2074 / 4092 = 50.684261975
x3 = 100 * 2094 / 4092 = 51.173020528

mu = (51.612903226 + 50.684261975 + 51.173020528) / 3
   = 51.156728576
SD = sqrt([(51.612903226 - 51.156728576)^2
         + (50.684261975 - 51.156728576)^2
         + (51.173020528 - 51.156728576)^2] / 3)
   = sqrt(0.143861809425)
   = 0.379291193 -> 0.38 pp
mu +/- SD = 51.16 +/- 0.38
```

### Cosmos Policy

Source: [evaluation records](results/cosmos-policy/results/libero_para_3seeds/cosmos_libero_para_3seeds_20260718_054146).

```text
x1 = 100 * 2893 / 4092 = 70.698924731
x2 = 100 * 2870 / 4092 = 70.136852395
x3 = 100 * 2892 / 4092 = 70.674486804

mu = (70.698924731 + 70.136852395 + 70.674486804) / 3
   = 70.503421310
SD = sqrt([(70.698924731 - 70.503421310)^2
         + (70.136852395 - 70.503421310)^2
         + (70.674486804 - 70.503421310)^2] / 3)
   = sqrt(0.067285920091)
   = 0.259395297 -> 0.26 pp
mu +/- SD = 70.50 +/- 0.26
```

## 6. The two OpenPI standard deviation corrections

Both calculations use the Clean scores 96.65, 96.70, and 96.20.

First, the population standard deviation of Clean Average is 0.224845626, so **96.52 ± 0.62 becomes 96.52 ± 0.22**.

Second, compute Clean-to-Para drop by subtracting the corresponding Para score from each Clean score, then take the population standard deviation of the three differences.

```text
d_r     = Clean_r - Para_r
mean_d  = (d1 + d2 + d3) / 3
SD_drop = sqrt([(d1 - mean_d)^2 + (d2 - mean_d)^2 + (d3 - mean_d)^2] / 3)
```

### OpenPI Clean-to-Para drop

Inputs: Clean scores from Section 3 and Para success counts from Section 5.

```text
x1 = 96.650000000 - 100 * 2919 / 4092 = 25.315689150
x2 = 96.700000000 - 100 * 2921 / 4092 = 25.316813294
x3 = 96.200000000 - 100 * 2916 / 4092 = 24.939002933

mu = (25.315689150 + 25.316813294 + 24.939002933) / 3
   = 25.190501792
SD = sqrt([(25.315689150 - 25.190501792)^2
         + (25.316813294 - 25.190501792)^2
         + (24.939002933 - 25.190501792)^2] / 3)
   = sqrt(0.031626048798)
   = 0.177837141 -> 0.18 pp
mu +/- SD = 25.19 +/- 0.18
```

Therefore, **25.19 ± 0.12 becomes 25.19 ± 0.18 pp**. The OpenPI Para success rate remains **71.33 ± 0.05**. The drop standard deviation cannot be obtained by subtracting the two standard deviations or by adding their squares without accounting for the paired covariance.

## 7. Recalculation code

The following code recalculates the means and standard deviations from the run scores listed above. Inputs are retained to 12 decimal places.

```python
from statistics import mean, pstdev

# population standard deviation; equivalent to numpy.std(..., ddof=0)
scores = {
    ('Clean', 'OpenPI'): [96.650000000000, 96.700000000000, 96.200000000000],
    ('Clean', 'UnifoLM-VLA-0'): [98.000000000000, 97.850000000000, 97.900000000000],
    ('Clean', 'Xiaomi-Robotics-0'): [97.500000000000, 98.300000000000, 98.500000000000],
    ('Clean', 'FastWAM'): [96.550000000000, 96.350000000000, 96.750000000000],
    ('Clean', 'Cosmos Policy'): [97.950000000000, 97.400000000000, 97.550000000000],
    ('Plus', 'OpenPI'): [85.006666666667, 85.755000000000, 85.160000000000],
    ('Plus', 'UnifoLM-VLA-0'): [78.440650871903, 79.159783643524, 78.732390333433],
    ('Plus', 'Xiaomi-Robotics-0'): [73.773458852043, 73.857253194805, 74.086813317385],
    ('Plus', 'GR00T-N1.7'): [74.936666666667, 74.898333333333, 75.513333333333],
    ('Plus', 'FastWAM'): [70.184064293653, 70.466723099620, 71.404322835464],
    ('Plus', 'Cosmos Policy'): [80.793262173447, 80.255981858277, 80.375553285687],
    ('Para', 'OpenPI'): [71.334310850440, 71.383186705767, 71.260997067449],
    ('Para', 'UnifoLM-VLA-0'): [81.891495601173, 82.917888563050, 81.915933528837],
    ('Para', 'Xiaomi-Robotics-0'): [75.708699902248, 75.928641251222, 75.537634408602],
    ('Para', 'GR00T-N1.7'): [75.537634408602, 75.122189638319, 72.116324535679],
    ('Para', 'FastWAM'): [51.612903225806, 50.684261974585, 51.173020527859],
    ('Para', 'Cosmos Policy'): [70.698924731183, 70.136852394917, 70.674486803519],
}

for (dataset, model), values in scores.items():
    print(dataset, model, f"{mean(values):.2f} +/- {pstdev(values):.2f}")

drop = [c - p for c, p in zip(scores[("Clean", "OpenPI")],
                             scores[("Para", "OpenPI")])]
print("OpenPI drop", f"{mean(drop):.2f} +/- {pstdev(drop):.2f}")
```

See [README Results](README.md#results) for the statistical definition. Section 1 lists the paper-reported results; Sections 3–6 show calculations using the inputs listed in this document.
