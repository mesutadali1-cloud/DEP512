"""
Part A - Analysis of a SDOF system using the Newmark-beta method

    m*x'' + c*x' + k*x = p(t)            (external excitation)
    m*x'' + c*x' + k*x = -m*xg''(t)      (ground motion)

Usage:
    python newmark_sdof.py                      # problems 1 and 2
    python newmark_sdof.py ground_acc.txt       # use a specific ground record

The ground acceleration file must contain one acceleration value per line
(m/s^2, dt = 0.005 s). If it has two columns (t, a), the second is used.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
# Newmark-beta function
# ----------------------------------------------------------------------------
def newmark(m, c, k, p, dt, beta, gamma, x0, v0):
    """Linear SDOF time integration with the Newmark-beta family.

    Parameters
    ----------
    m, c, k : mass, damping coefficient, stiffness
    p       : load vector p(t_i), i = 0..N-1 (sampled with step dt)
    dt      : time step
    beta, gamma : Newmark parameters
                  (1/4, 1/2) average acceleration (unconditionally stable)
                  (1/6, 1/2) linear acceleration  (stable if dt/Tn <= 0.551)
                  (0,   1/2) central difference  (stable if dt/Tn <= 1/pi)
    x0, v0  : initial displacement and velocity

    Returns
    -------
    x, v, a : displacement, velocity and acceleration vectors
    """
    p = np.asarray(p, dtype=float)
    n = len(p)
    x = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)

    # initial conditions; initial acceleration from equation of motion
    x[0] = x0
    v[0] = v0
    a[0] = (p[0] - c * v0 - k * x0) / m

    # effective "mass" - written in acceleration form so that beta = 0
    # (explicit central difference) is also allowed
    m_eff = m + gamma * dt * c + beta * dt**2 * k

    for i in range(n - 1):
        # predictors (the parts that depend only on step i)
        x_pred = x[i] + dt * v[i] + (0.5 - beta) * dt**2 * a[i]
        v_pred = v[i] + (1.0 - gamma) * dt * a[i]

        # equilibrium at t_{i+1}:  m*a + c*(v_pred + gamma*dt*a) + k*(x_pred + beta*dt^2*a) = p
        a[i + 1] = (p[i + 1] - c * v_pred - k * x_pred) / m_eff

        # correctors
        x[i + 1] = x_pred + beta * dt**2 * a[i + 1]
        v[i + 1] = v_pred + gamma * dt * a[i + 1]

    return x, v, a


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def system_from_wn_xi(m, wn, xi):
    """k and c giving the requested natural frequency and damping ratio."""
    k = m * wn**2
    c = 2.0 * xi * m * wn
    return k, c


def exact_harmonic(m, c, k, P, wbar, t):
    """Closed-form response to p = P sin(wbar t) with x0 = v0 = 0 (for checking)."""
    wn = np.sqrt(k / m)
    xi = c / (2 * m * wn)
    wd = wn * np.sqrt(1 - xi**2)
    r = wbar / wn
    xst = P / k
    den = (1 - r**2) ** 2 + (2 * xi * r) ** 2
    C = xst * (1 - r**2) / den
    D = xst * (-2 * xi * r) / den
    # steady state:  C sin(wbar t) + D cos(wbar t)
    # transient   :  e^{-xi wn t} (A sin(wd t) + B cos(wd t)) with x(0)=v(0)=0
    B = -D
    A = (xi * wn * B - C * wbar) / wd
    return (np.exp(-xi * wn * t) * (A * np.sin(wd * t) + B * np.cos(wd * t))
            + C * np.sin(wbar * t) + D * np.cos(wbar * t))


def plot_results(t, x, v, a, k, c, title, fname, accel_label="Acceleration [m/s²]"):
    fs = k * x          # spring force
    fd = c * v          # damper force

    fig, ax = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    data = [(x, "Displacement [m]"), (v, "Velocity [m/s]"), (a, accel_label),
            (fs, "Spring force  k·x [N]"), (fd, "Damper force  c·v [N]")]
    for axi, (y, lab) in zip(ax, data):
        axi.plot(t, y, lw=0.9)
        axi.set_ylabel(lab)
        axi.grid(alpha=0.3)
    ax[-1].set_xlabel("Time [s]")
    fig.suptitle(title + " — time histories")
    fig.tight_layout()
    fig.savefig(fname + "_time_history.png", dpi=150)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].plot(x, fs, lw=0.9)
    ax[0].set_xlabel("Displacement [m]")
    ax[0].set_ylabel("Spring force [N]")
    ax[0].set_title("Displacement vs spring force")
    ax[1].plot(x, fd, lw=0.9, color="tab:red")
    ax[1].set_xlabel("Displacement [m]")
    ax[1].set_ylabel("Damper force [N]")
    ax[1].set_title("Displacement vs damper force")
    for axi in ax:
        axi.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(fname + "_force_disp.png", dpi=150)


def load_ground_motion(path):
    data = np.loadtxt(path, ndmin=2)
    return data[:, -1] if data.shape[1] > 1 else data[:, 0]


def compare_parameters(m, c, k, p, dt, t, title, fname, x_ref=None):
    """Run the same problem with several (beta, gamma) pairs and overlay x(t)."""
    cases = [(1 / 4, 1 / 2, "Average acc. (β=1/4, γ=1/2)"),
             (1 / 6, 1 / 2, "Linear acc. (β=1/6, γ=1/2)"),
             (0.0, 1 / 2, "Central diff. (β=0, γ=1/2)"),
             (0.3025, 0.6, "Damped (β=0.3025, γ=0.6)")]
    plt.figure(figsize=(10, 4.5))
    if x_ref is not None:
        plt.plot(t, x_ref, "k", lw=2.5, alpha=0.35, label="Exact")
    print(f"\n  {title}: peak |x| for different Newmark parameters")
    for beta, gamma, lab in cases:
        x, _, _ = newmark(m, c, k, p, dt, beta, gamma, 0.0, 0.0)
        plt.plot(t, x, lw=0.9, label=lab)
        line = f"    {lab:32s} max|x| = {np.max(np.abs(x)):.6e} m"
        if x_ref is not None:
            line += f"   max error = {np.max(np.abs(x - x_ref)):.2e} m"
        print(line)
    plt.xlabel("Time [s]")
    plt.ylabel("Displacement [m]")
    plt.title(title + " — effect of β, γ")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname + "_beta_gamma.png", dpi=150)


# ----------------------------------------------------------------------------
# Main part
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))

    # Newmark parameters (suggested: average acceleration)
    beta, gamma = 1 / 4, 1 / 2

    # System: wn = 2*pi rad/s (Tn = 1 s), xi = 5 %
    m = 1.0                         # kg
    wn = 2 * np.pi                  # rad/s
    xi = 0.05
    k, c = system_from_wn_xi(m, wn, xi)
    print(f"System: m = {m} kg, k = {k:.4f} N/m, c = {c:.4f} N.s/m "
          f"(wn = {wn:.4f} rad/s, Tn = {2*np.pi/wn:.2f} s, xi = {xi})")

    # ---------------- Problem 1: p(t) = P sin(wbar t) ----------------
    P, wbar = 2.0, np.pi / 2
    dt1, t_end = 0.01, 20.0
    t1 = np.arange(0.0, t_end + dt1 / 2, dt1)
    p1 = P * np.sin(wbar * t1)
    x0, v0 = 0.0, 0.0

    x1, v1, a1 = newmark(m, c, k, p1, dt1, beta, gamma, x0, v0)
    x1_exact = exact_harmonic(m, c, k, P, wbar, t1)
    print(f"\nProblem 1: P = {P}, wbar = {wbar:.4f} rad/s, dt = {dt1} s")
    print(f"  max|x| = {np.max(np.abs(x1)):.6e} m,  max error vs exact = "
          f"{np.max(np.abs(x1 - x1_exact)):.2e} m")
    print(f"  steady-state amplitude (theory) = "
          f"{P/k/np.sqrt((1-(wbar/wn)**2)**2 + (2*xi*wbar/wn)**2):.6e} m")

    plot_results(t1, x1, v1, a1, k, c, "Problem 1: p(t) = 2 sin(πt/2)",
                 os.path.join(out_dir, "P1"))
    compare_parameters(m, c, k, p1, dt1, t1, "Problem 1",
                       os.path.join(out_dir, "P1"), x_ref=x1_exact)

    # ---------------- Problem 2: ground acceleration ----------------
    dt2 = 0.005
    gm_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(out_dir, "ground_acc.txt")
    if os.path.exists(gm_path):
        ag = load_ground_motion(gm_path)
        gm_name = os.path.basename(gm_path)
    else:
        print(f"\n[WARNING] '{gm_path}' not found - using a SYNTHETIC test record. "
              "Put the homework data in ground_acc.txt (or pass its path).")
        rng = np.random.default_rng(0)
        tt = np.arange(0, 20, dt2)
        ag = (rng.standard_normal(tt.size) * 2.0
              * (tt / 2) ** 2 * np.exp(-(tt / 2) + 1) / 4)   # enveloped white noise
        gm_name = "synthetic record"

    t2 = np.arange(len(ag)) * dt2
    p2 = -m * ag
    x2, v2, a2 = newmark(m, c, k, p2, dt2, beta, gamma, 0.0, 0.0)
    a2_abs = a2 + ag                  # total (absolute) acceleration

    print(f"\nProblem 2: {gm_name}, N = {len(ag)}, dt = {dt2} s, PGA = {np.max(np.abs(ag)):.4f} m/s²")
    print(f"  max|x| = {np.max(np.abs(x2)):.6e} m,  max|v| = {np.max(np.abs(v2)):.6e} m/s")
    print(f"  max|a_rel| = {np.max(np.abs(a2)):.6e},  max|a_abs| = {np.max(np.abs(a2_abs)):.6e} m/s²")
    print(f"  max spring force = {np.max(np.abs(k*x2)):.6e} N, "
          f"max damper force = {np.max(np.abs(c*v2)):.6e} N")

    plot_results(t2, x2, v2, a2, k, c, f"Problem 2: ground motion ({gm_name})",
                 os.path.join(out_dir, "P2"), accel_label="Relative acc. [m/s²]")
    compare_parameters(m, c, k, p2, dt2, t2, "Problem 2", os.path.join(out_dir, "P2"))

    plt.show()
