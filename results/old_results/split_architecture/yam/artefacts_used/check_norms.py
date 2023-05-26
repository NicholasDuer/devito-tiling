import numpy as np

f = open("norms.txt", "r")
lines = f.readlines()

norm = lines[0]

for norm_1 in lines:
    for norm_2 in lines:
        assert np.isclose(float(norm_1), float(norm_2), atol=1e-4, rtol=0)

print("Asserted!, Norm: " + norm)