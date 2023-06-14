import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def plot():
    # Define the ranges for s and T
    s_values = np.linspace(2, 8, 5000)
    T_values = np.linspace(4, 32, 5000)
    s, T = np.meshgrid(s_values, T_values)

    # Compute the corresponding function values
    Z = calc_number_points(s, T)

    # Create the surface plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(s, T, Z, cmap='Reds', antialiased=False)

    # Set labels and title
    ax.set_xlabel('Space Order, s')
    ax.set_ylabel('Time Tile Height, T')
    ax.set_zlabel('% Increase in Points Processed')
    ax.set_zlim(np.nanmin(Z), np.nanmax(Z))
    ax.grid(False)
    plt.savefig("extra_points_count")

def calc_number_points(s, T):
    tM = 256
    xM = 256
    yM = 512
    zM = 512
    a = zM * s**2
    b = zM * s * (-xM - yM - 2 * s * (T - 1))
    c = zM * (xM * yM + (xM + yM) * (T - 1) * s + s**2 * (T - 1)**2)
    total = tM * (a/6 * (T - 1) * (2*T - 1) + b/2 * (T - 1) + c)
    return ((total / (tM * xM * yM * zM)) - 1) * 100

font = {
    'size'   : 12}

matplotlib.rc('font', **font)
plot()
