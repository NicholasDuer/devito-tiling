import numpy as np

f = open("norms.txt", "r")
lines = f.readlines()

overlapped_norm = float(lines[0])
standard_norm = float(lines[1])

assert np.isclose(overlapped_norm, standard_norm, atol=1e-4, rtol=0)
print("Asserted!, Norm: " + str(overlapped_norm))