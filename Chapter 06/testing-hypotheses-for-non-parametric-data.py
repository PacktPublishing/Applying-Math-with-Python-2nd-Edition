from scipy import stats
from numpy.random import default_rng
rng = default_rng(12345)


sample_A = rng.uniform(2.5, 3.5, size=25)
sample_B = rng.uniform(3.0, 4.4, size=25)
sample_C = rng.uniform(3.1, 4.5, size=25)

significance = 0.05

statistic, p_value = stats.kruskal(sample_A, sample_B, sample_C)
print(f"Statistic: {statistic}, p value: {p_value}")
# Statistic: 40.22214736842102, p value: 1.8444703308682906e-09

if p_value <= significance:
    print("There are differences between population medians")
else:
    print("Accept H0: all medians equal")
# There are differences between population medians

_, p_A_B = stats.ranksums(sample_A, sample_B)
_, p_A_C = stats.ranksums(sample_A, sample_C)
_, p_B_C = stats.ranksums(sample_B, sample_C)

if p_A_B <= significance:
    print("Significant differences between A and B, p value", p_A_B)
# Significant differences between A and B, p value 1.0035366080480683e-07

if p_A_C <= significance:
    print("Significant differences between A and C, p value", p_A_C)
# Significant differences between A and C, p value 2.428534673701913e-08

if p_B_C <= significance:
    print("Significant differences between B and C, p value", p_B_C)
else:
    print("No significant differences between B and C, p value", p_B_C)
# No significant differences between B and C, p value 0.3271631660572756
