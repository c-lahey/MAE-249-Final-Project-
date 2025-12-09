import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# ---- Physical parameters ----
E = 1250000      # Pa
d = 0.003          # m
I = (np.pi * (d/2)**4) / 4.0
l = 0.004          # m
K = E * I / l

N  = 12      # number of links
B  = 0.015  # nominal magnetic field strength (used as initial guess)
mu = 0.0005 # magnetic dipole strength

# Axial polarization
phi = np.zeros(N)  # polarization along links

# Collect parameters
params = dict(
    E=E, d=d, I=I, l=l, K=K,
    N=N, B=B, mu=mu, phi=phi
)

# Initial guesses
psi_init   = -np.pi/2       # initial guess for field orientation

# ---- Choose a desired tip position ----

x_des = 0.0322
y_des =  -0.0319

# Better initial guess for joint angles from desired tip direction
theta0_hat = np.arctan2(y_des, x_des)
theta0 = theta0_hat * np.ones(N)


# ============================================================
#   Energy, equilibrium, and backbone geometry
# ============================================================

def catheter_energy(theta, psi, params):
    """Total potential energy: bending + magnetic."""
    theta = np.asarray(theta)
    K   = params["K"]
    B   = params["B"]
    mu  = params["mu"]
    phi = params["phi"]

    # Bending energy
    PI_bend = 0.5 * K * np.sum(theta**2)

    # Magnetic energy
    abs_angles = np.cumsum(theta)
    arg = -(phi + abs_angles) + psi
    PI_mag = -mu * B * np.sum(np.cos(arg))

    return PI_bend + PI_mag


def solve_equilibrium(theta0, psi, params, tol=1e-6, maxiter=200):
    """
    Minimize energy w.r.t. theta using scipy.optimize.minimize.

    Looser tolerances to speed up repeated calls.
    """
    fun = lambda th: catheter_energy(th, psi, params)
    res = minimize(
        fun,
        x0=np.asarray(theta0),
        method="BFGS",
        options=dict(gtol=tol, maxiter=maxiter, disp=False)
    )
    return res.x


def backbone_xy(abs_angles, l):
    """Compute node coordinates from absolute angles."""
    abs_angles = np.asarray(abs_angles)
    N = abs_angles.size
    x = np.zeros(N+1)
    y = np.zeros(N+1)
    for i in range(1, N+1):
        a = abs_angles[i-1]
        x[i] = x[i-1] + l * np.sin(a)
        y[i] = y[i-1] - l * np.cos(a)
    return x, y


# ---------- Forward map: field -> tip position ----------

def configuration_from_field(field, theta0, params,
                             tol=1e-6, maxiter=200):
    """
    Given field = [B, psi], solve for equilibrium configuration and return
    (x_nodes, y_nodes, abs_angles, theta_eq).
    """
    B_val, psi_val = field
    p_local = params.copy()
    p_local["B"] = B_val

    theta_eq = solve_equilibrium(theta0, psi_val, p_local,
                                 tol=tol, maxiter=maxiter)
    abs_angles = np.cumsum(theta_eq)
    x, y = backbone_xy(abs_angles, p_local["l"])
    return x, y, abs_angles, theta_eq


def tip_position_from_field(field, theta0, params,
                            tol=1e-6, maxiter=200):
    """Return (x_tip, y_tip) for given field parameters."""
    x, y, _, _ = configuration_from_field(field, theta0, params,
                                          tol=tol, maxiter=maxiter)
    return x[-1], y[-1]


# ============================================================
#   Inverse kinematics: desired tip -> optimal field
# ============================================================

def inverse_kinematics_tip(x_target, y_target,
                           theta0, params,
                           B_init=None, psi_init=None,
                           n_starts=10,
                           random_starts=True,
                           seed=0,
                           inner_tol=1e-6,
                           inner_maxiter=200):
    """
    Multi-start optimization over [B, psi] to match tip position to (x_target, y_target).

    Uses a warm-start for theta inside each local run so that each field
    evaluation reuses the previous equilibrium configuration instead of
    restarting from the original theta0.
    """

    if B_init is None:
        B_init = params["B"]
    if psi_init is None:
        psi_init = -np.pi/2

    theta0 = np.asarray(theta0)

    # Bounds for [B, psi]
    bounds = [
        (0.0, 0.5),      # B in Tesla
        (-np.pi, np.pi)   # psi in radians
    ]

    # ---- Build list of starting points ----
    starts = []
    starts.append(np.array([B_init, psi_init], dtype=float))  # main guess

    n_extra = max(n_starts - 1, 0)

    if n_extra > 0:
        if random_starts:
            rng = np.random.default_rng(seed)
            for _ in range(n_extra):
                B0  = rng.uniform(bounds[0][0], bounds[0][1])
                psi0 = rng.uniform(bounds[1][0], bounds[1][1])
                starts.append(np.array([B0, psi0], dtype=float))
        else:
            # Deterministic grid
            n_B = int(np.sqrt(n_extra))
            n_psi = int(np.ceil(n_extra / max(n_B, 1)))
            B_vals = np.linspace(bounds[0][0], bounds[0][1], n_B)
            psi_vals = np.linspace(bounds[1][0], bounds[1][1], n_psi)
            for B0 in B_vals:
                for psi0 in psi_vals:
                    if len(starts) >= n_starts:
                        break
                    if np.allclose([B0, psi0], [B_init, psi_init]):
                        continue
                    starts.append(np.array([B0, psi0], dtype=float))
                if len(starts) >= n_starts:
                    break

    # Helper for a single local optimization, with warm-start on theta
    def run_local(start_field):
        theta_guess = theta0.copy()

        def objective(field):
            nonlocal theta_guess
            B_val, psi_val = field
            p_local = params.copy()
            p_local["B"] = B_val

            # Inner equilibrium solve, warm-starting from previous theta_guess
            theta_eq = solve_equilibrium(theta_guess, psi_val, p_local,
                                         tol=inner_tol, maxiter=inner_maxiter)
            theta_guess = theta_eq

            abs_angles = np.cumsum(theta_eq)
            x, y = backbone_xy(abs_angles, p_local["l"])

            dx = x[-1] - x_target
            dy = y[-1] - y_target
            return dx*dx + dy*dy

        res = minimize(
            objective,
            np.asarray(start_field, dtype=float),
            method="L-BFGS-B",
            bounds=bounds,
            options=dict(eps=1e-3, gtol=1e-6, maxiter=200, disp=False)
        )
        return res, theta_guess

    # ---- Run multi-start optimization ----
    best_res = None
    best_val = np.inf
    best_theta_guess = theta0.copy()

    for start in starts:
        res, theta_last = run_local(start)
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_res = res
            best_theta_guess = theta_last.copy()

    # Fallback: if *everything* failed, do a single run from [B_init, psi_init]
    if best_res is None:
        best_res, best_theta_guess = run_local([B_init, psi_init])

    field_opt = best_res.x
    B_opt, psi_opt = field_opt

    # One final equilibrium solve at the optimal field,
    # warm-starting from the best theta we saw in that neighborhood.
    p_local = params.copy()
    p_local["B"] = B_opt
    theta_opt = solve_equilibrium(best_theta_guess, psi_opt, p_local,
                                  tol=inner_tol, maxiter=inner_maxiter)
    abs_angles_opt = np.cumsum(theta_opt)
    x_opt, y_opt = backbone_xy(abs_angles_opt, p_local["l"])

    return field_opt, x_opt, y_opt, abs_angles_opt, theta_opt, best_res


# ============================================================
#   Run IK and visualize
# ============================================================

field_opt, x_nodes, y_nodes, abs_angles, theta_eq, res_field = inverse_kinematics_tip(
    x_des, y_des,
    theta0, params,
    B_init=B, psi_init=psi_init,
    n_starts=10,          # fewer starts; adjust if needed
    random_starts=True,
    seed=0,
    inner_tol=1e-6,
    inner_maxiter=200
)

B_opt, psi_opt = field_opt
x_tip = x_nodes[-1]
y_tip = y_nodes[-1]

print("Optimization success:", res_field.success)
print("Optimal B   [T]:", B_opt)
print("Optimal psi [deg]:", np.degrees(psi_opt))
print("Tip position (x, y) [m]:", x_tip, y_tip)
print("Target      (x, y) [m]:", x_des, y_des)
print("Tip error [m]:", np.sqrt((x_tip - x_des)**2 + (y_tip - y_des)**2))


# ---- Visualization ----

abs_polarization_angles = phi + abs_angles
xc = 0.5 * (x_nodes[:-1] + x_nodes[1:])
yc = 0.5 * (y_nodes[:-1] + y_nodes[1:])

dip_vx = np.sin(abs_polarization_angles)
dip_vy = -np.cos(abs_polarization_angles)

# Field direction from optimal psi
Bdir = np.array([np.sin(psi_opt), -np.cos(psi_opt)])

fig, ax = plt.subplots()

# Catheter backbone
ax.plot(x_nodes, y_nodes, '-k', linewidth=2)
ax.plot(x_nodes, y_nodes, 'ko', markersize=4)

# Dipoles
Lvec = 0.2 * l
ax.quiver(xc, yc, Lvec*dip_vx, Lvec*dip_vy, angles='xy', scale_units='xy', scale=1,
          color=[0.85, 0.1, 0.1])

# Background field vectors
xg, yg = np.meshgrid(
    np.linspace(x_nodes.min()-0.02, x_nodes.max()+0.02, 10),
    np.linspace(y_nodes.min()-0.02, y_nodes.max()+0.02, 10)
)
Lfield = 0.005
ax.quiver(xg, yg, Lfield*Bdir[0]*np.ones_like(xg),
                    Lfield*Bdir[1]*np.ones_like(yg),
          angles='xy', scale_units='xy', scale=1,
          color=[0.2, 0.4, 0.9])

# Desired tip position
ax.plot(x_des, y_des, 'rx', markersize=8, mew=2, label='Desired tip')

# Achieved tip position
ax.plot(x_tip, y_tip, 'g+', markersize=10, mew=2, label='Achieved tip')

# Formatting
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_title('Inverse-kinematic control of magnetic catheter (Python + scipy)')
ax.grid(True)

final_angle_deg = np.degrees(abs_angles[-1])
ax.text(x_nodes.max()+0.01, y_nodes.min()-0.01,
        f'Final link angle: {final_angle_deg:.2f}°\n'
        f'B = {B_opt:.4f} T\n'
        f'psi = {np.degrees(psi_opt):.1f}°',
        fontsize=9)

ax.legend()
plt.show()
