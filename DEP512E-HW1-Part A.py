import os
import numpy as np
import matplotlib.pyplot as plt


os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Newmark parameters
beta = 1 / 4
gamma = 1 / 2

#  wn = 2*pi rad/s, xi = 5 %
m = 1.0
wn = 2 * np.pi
xi = 0.05
k = m * wn ** 2
c = 2 * xi * m * wn

def newmark(m, c, k, p, dt, beta, gamma, x0, v0):
    n = len(p)
    x = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)

    # initial conditions
    x[0] = x0
    v[0] = v0
    a[0] = (p[0] - c * v0 - k * x0) / m

    # effective stiffness and constants
    k_hat = k + gamma / (beta * dt) * c + m / (beta * dt**2)
    A = m / (beta * dt) + gamma / beta * c
    B = m / (2 * beta) + dt * (gamma / (2 * beta) - 1) * c

    for i in range(n - 1):
        dp_hat = (p[i + 1] - p[i]) + A * v[i] + B * a[i]
        dx = dp_hat / k_hat
        dv = gamma / (beta * dt) * dx - gamma / beta * v[i] + dt * (1 - gamma / (2 * beta)) * a[i]
        da = dx / (beta * dt**2) - v[i] / (beta * dt) - a[i] / (2 * beta)

        x[i + 1] = x[i] + dx
        v[i + 1] = v[i] + dv
        a[i + 1] = a[i] + da

    return x, v, a
# Problem 1: p(t) = P*sin(wbar*t)
P = 2.0
wbar = np.pi / 2
dt = 0.01
t = np.arange(0, 20 + dt, dt)
p = P * np.sin(wbar * t)

x, v, a = newmark(m, c, k, p, dt, beta, gamma, 0, 0)
plt.figure()
plt.plot(t, x)
plt.title("Problem 1 - Displacement")
plt.xlabel("Time [s]")
plt.ylabel("Displacement [m]")

plt.figure()
plt.plot(t, v)
plt.title("Problem 1 - Velocity")
plt.xlabel("Time [s]")
plt.ylabel("Velocity [m/s]")

plt.figure()
plt.plot(t, a)
plt.title("Problem 1 - Acceleration")
plt.xlabel("Time [s]")
plt.ylabel("Acceleration [m/s^2]")

fs = k * x  # spring force
fd = c * v  # damper force

plt.figure()
plt.plot(t, k * x)
plt.title("Problem 1 - Spring Force")
plt.xlabel("Time [s]")
plt.ylabel("Spring force [N]")

plt.figure()
plt.plot(x, k * x)
plt.title("Problem 1 - Spring Force vs Displacement")
plt.xlabel("Displacement [m]")
plt.ylabel("Spring force [N]")


plt.figure()
plt.plot(t, c * v)
plt.title("Problem 1 - Damper Force")
plt.xlabel("Time [s]")
plt.ylabel("Damper force [N]")

plt.figure()
plt.plot(x, c * v)
plt.title("Problem 1 - Damper Force vs Displacement")
plt.xlabel("Displacement [m]")
plt.ylabel("Damper force [N]")


plt.show()

 #Problem 2: ground acceleration (m/s^2), dt = 0.005 s
dt = 0.005
ag = np.loadtxt("ground_acc.txt")
t = np.arange(len(ag)) * dt
p = -m * ag

x, v, a = newmark(m, c, k, p, dt, beta, gamma, 0, 0)
plt.figure()
plt.plot(t, x)
plt.title("Problem 2 - Displacement")
plt.xlabel("Time [s]")
plt.ylabel("Displacement [m]")

plt.figure()
plt.plot(t, v)
plt.title("Problem 2 - Velocity")
plt.xlabel("Time [s]")
plt.ylabel("Velocity [m/s]")

plt.figure()
plt.plot(t, a)
plt.title("Problem 2 - Acceleration")
plt.xlabel("Time [s]")
plt.ylabel("Acceleration [m/s^2]")

fs = k * x  # spring force
fd = c * v  # damper force

plt.figure()
plt.plot(t, k * x)
plt.title("Problem 2 - Spring Force")
plt.xlabel("Time [s]")
plt.ylabel("Spring force [N]")

plt.figure()
plt.plot(x, k * x)
plt.title("Problem 2 - Spring Force vs Displacement")
plt.xlabel("Displacement [m]")
plt.ylabel("Spring force [N]")


plt.figure()
plt.plot(t, c * v)
plt.title("Problem 2 - Damper Force")
plt.xlabel("Time [s]")
plt.ylabel("Damper force [N]")

plt.figure()
plt.plot(x, c * v)
plt.title("Problem 2 - Damper Force vs Displacement")
plt.xlabel("Displacement [m]")
plt.ylabel("Damper force [N]")

plt.show()



