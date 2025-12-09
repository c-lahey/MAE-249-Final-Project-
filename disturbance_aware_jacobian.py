import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from catheter_IK3 import inverse_kinematics_tip

# ============================================================
#   Physical parameters (model vs. true plant)
# ============================================================

# "Nominal" model parameters (used by controller / Jacobian)
E_nom = 1_250_000.0   # Pa
d_nom = 0.004         # m
I_nom = (np.pi * (d_nom/2)**4) / 4.0
l_nom = 0.004         # m
K_nom = E_nom * I_nom / l_nom

N  = 12               # number of links
B_nom = 0.015         # nominal B magnitude
mu_nom = 0.005       # magnetic dipole strength
phi = np.zeros(N)     # axial polarization

params_model = dict(
    E=E_nom, d=d_nom, I=I_nom, l=l_nom, K=K_nom,
    N=N, B=B_nom, mu=mu_nom, phi=phi
)

# "True" plant parameters
E_true = E_nom
d_true = d_nom
I_true = (np.pi * (d_true/2)**4) / 4.0
l_true = l_nom
K_true = E_true * I_true / l_true
B_true_nom = B_nom
mu_true = mu_nom

params_true = dict(
    E=E_true, d=d_true, I=I_true, l=l_true, K=K_true,
    N=N, B=B_true_nom, mu=mu_true, phi=phi
)

# ============================================================
#   Geometry helper
# ============================================================

def backbone_xy(abs_angles, l):
    """
    Compute backbone node coordinates from absolute link angles.
    Angles measured w.r.t. downward vertical.
    """
    abs_angles = np.asarray(abs_angles)
    N = abs_angles.size
    x = np.zeros(N+1)
    y = np.zeros(N+1)
    for i in range(1, N+1):
        a = abs_angles[i-1]
        x[i] = x[i-1] + l * np.sin(a)
        y[i] = y[i-1] - l * np.cos(a)
    return x, y

# ============================================================
#   Energy, equilibrium, backbone geometry
# ============================================================

def catheter_energy(theta, psi, params):
    theta = np.asarray(theta)
    K   = params["K"]
    B   = params["B"]
    mu  = params["mu"]
    phi = params["phi"]

    # Bending energy
    PI_bend = 0.5 * K * np.sum(theta**2)

    # Magnetic potential
    abs_angles = np.cumsum(theta)
    arg = -(phi + abs_angles) + psi
    PI_mag = -mu * B * np.sum(np.cos(arg))

    return PI_bend + PI_mag


def catheter_energy_true(theta, psi, params, F_tip=0.0, alpha_rel=0.0):
    """
    True plant energy = model energy + tip force disturbance.
    """
    theta = np.asarray(theta)

    # Nominal part
    E_nom_val = catheter_energy(theta, psi, params)

    if F_tip == 0.0:
        return E_nom_val

    # Absolute angles and tip orientation
    abs_angles = np.cumsum(theta)
    theta_tip = abs_angles[-1]

    # Tip position from backbone geometry
    l = params["l"]
    x_nodes, y_nodes = backbone_xy(abs_angles, l)
    x_tip = x_nodes[-1]
    y_tip = y_nodes[-1]
    r_tip = np.array([x_tip, y_tip])

    # Force direction in global frame
    psi_F = theta_tip + alpha_rel
    Fx = -F_tip * np.sin(psi_F)
    Fy =  F_tip * np.cos(psi_F)
    F_vec = np.array([Fx, Fy])

    # Disturbance potential: Π_dist = -F · r_tip
    PI_dist = -np.dot(F_vec, r_tip)

    return E_nom_val + PI_dist


def solve_equilibrium(theta0, psi, params, tol=1e-6, maxiter=200):
    """Minimize energy w.r.t. theta using BFGS (nominal model)."""
    fun = lambda th: catheter_energy(th, psi, params)
    res = minimize(
        fun,
        x0=np.asarray(theta0),
        method="BFGS",
        options=dict(gtol=tol, maxiter=maxiter, disp=False)
    )
    return res.x


def solve_equilibrium_true(theta0, psi, params, F_tip=0.0, alpha_rel=0.0,
                           tol=1e-6, maxiter=200):
    """Minimize true plant energy (with tip force disturbance)."""
    fun = lambda th: catheter_energy_true(th, psi, params, F_tip, alpha_rel)
    res = minimize(
        fun,
        x0=np.asarray(theta0),
        method="BFGS",
        options=dict(gtol=tol, maxiter=maxiter, disp=False)
    )
    return res.x


def Bvec_to_B_psi(Bx, By, B_min=1e-6):
    """
    Convert field vector (Bx, By) to (B, psi) with Bx = B sin(psi),
    By = -B cos(psi).
    """
    B = np.sqrt(Bx**2 + By**2)
    if B < B_min:
        B = B_min
    psi = np.arctan2(Bx, -By)
    return B, psi


def forward_equilibrium_BxBy(u, theta_guess, params,
                             tol=1e-6, maxiter=200):
    """
    Forward map for a given field u = [Bx, By] (nominal model).
    Returns tip position, backbone nodes, theta, abs_angles.
    """
    Bx, By = u
    B_val, psi_val = Bvec_to_B_psi(Bx, By)

    p_local = params.copy()
    p_local["B"] = B_val

    theta_eq = solve_equilibrium(theta_guess, psi_val, p_local,
                                 tol=tol, maxiter=maxiter)
    abs_angles = np.cumsum(theta_eq)
    x, y = backbone_xy(abs_angles, p_local["l"])
    p_tip = np.array([x[-1], y[-1]])
    return p_tip, x, y, theta_eq, abs_angles


def forward_equilibrium_BxBy_true(u, theta_guess, params,
                                  F_tip=0.0, alpha_rel=0.0,
                                  tol=1e-6, maxiter=200):
    """
    Forward map for given u = [Bx, By] on the *true* plant
    including a tip force disturbance.
    Returns tip position, backbone nodes, theta, abs_angles.
    """
    Bx, By = u
    B_val, psi_val = Bvec_to_B_psi(Bx, By)

    p_local = params.copy()
    p_local["B"] = B_val

    theta_eq = solve_equilibrium_true(theta_guess, psi_val, p_local,
                                      F_tip=F_tip, alpha_rel=alpha_rel,
                                      tol=tol, maxiter=maxiter)
    abs_angles = np.cumsum(theta_eq)
    x, y = backbone_xy(abs_angles, p_local["l"])
    p_tip = np.array([x[-1], y[-1]])
    return p_tip, x, y, theta_eq, abs_angles

# ============================================================
#   Numerical Jacobian (disturbance-aware, true plant)
# ============================================================

def numerical_jacobian_u_BxBy_true_forced(
    u,
    theta_guess,
    params_true,
    F_tip_hat,
    alpha_rel_hat,
    dBx=1e-4,
    dBy=1e-4
):
    """
    Jacobian J = ∂p_tip/∂u for the *true* plant under a fixed
    tip disturbance (F_tip_hat, alpha_rel_hat).

    Uses finite differences around the equilibrium of the true plant
    with that disturbance applied.
    """
    u = np.asarray(u, dtype=float)

    # Base point: equilibrium with (u, F_hat, alpha_hat)
    p0, _, _, theta0, _ = forward_equilibrium_BxBy_true(
        u,
        theta_guess,
        params_true,
        F_tip=F_tip_hat,
        alpha_rel=alpha_rel_hat
    )
    theta_base = theta0.copy()

    # Perturb Bx
    uBx = u.copy()
    uBx[0] += dBx
    pBx, _, _, thetaBx, _ = forward_equilibrium_BxBy_true(
        uBx,
        theta_base,
        params_true,
        F_tip=F_tip_hat,
        alpha_rel=alpha_rel_hat
    )
    dp_dBx = (pBx - p0) / dBx

    # Perturb By
    uBy = u.copy()
    uBy[1] += dBy
    pBy, _, _, thetaBy, _ = forward_equilibrium_BxBy_true(
        uBy,
        theta_base,
        params_true,
        F_tip=F_tip_hat,
        alpha_rel=alpha_rel_hat
    )
    dp_dBy = (pBy - p0) / dBy

    J = np.column_stack((dp_dBx, dp_dBy))   # shape (2,2)
    return J, thetaBy

# ============================================================
#   Magnetics-based tip force estimation (tip position only)
# ============================================================

def estimate_tip_force_from_tip(
    u,
    p_tip_meas,
    theta_init,
    params_true,
    F_init=0.01,
    alpha_init=0.0,
    F_bounds=(0.0, 0.5),
    alpha_bounds=(-np.pi, np.pi),
    tol=1e-6,
    maxiter=100
):
    """
    Estimate tip disturbance magnitude F_tip and relative direction alpha_rel
    from a measured TIP POSITION under a known magnetic field u = [Bx, By].

    Solves:
        min_{F, alpha} || p_tip_model(u, F, alpha) - p_tip_meas ||^2
    """
    p_tip_meas = np.asarray(p_tip_meas, dtype=float)

    def cost(z):
        F = z[0]
        alpha = z[1]

        if F < 0.0:
            # Penalize negative forces strongly
            return 1e6 + 1e3 * (-F)

        p_tip_pred, _, _, _, _ = forward_equilibrium_BxBy_true(
            u,
            theta_init,
            params_true,
            F_tip=F,
            alpha_rel=alpha,
            tol=1e-6,
            maxiter=200
        )

        err = p_tip_pred - p_tip_meas
        return float(np.dot(err, err))

    z0 = np.array([F_init, alpha_init])
    bounds = [F_bounds, alpha_bounds]

    res = minimize(
        cost,
        z0,
        method="L-BFGS-B",
        bounds=bounds,
        options=dict(ftol=tol, maxiter=maxiter, disp=False)
    )

    F_hat, alpha_hat = res.x

    # Clean up outputs
    F_hat = max(F_hat, 0.0)
    alpha_hat = (alpha_hat + np.pi) % (2 * np.pi) - np.pi  # wrap into [-pi, pi]

    return F_hat, alpha_hat, res

# ============================================================
#   Desired tip, feedforward field via IK (shared)
# ============================================================

x_des = 0.0322
y_des = -0.0319
p_des = np.array([x_des, y_des])

theta0_hat = np.arctan2(y_des, x_des)
theta_init_nom = theta0_hat * np.ones(N)

field_opt, _, _, _, _, _ = inverse_kinematics_tip(
    x_des, y_des,
    theta_init_nom, params_model,
    B_init=0.015, psi_init=-np.pi/2,
    n_starts=10
)

B_ff, psi_ff = field_opt
u0 = np.array([
    B_ff * np.sin(psi_ff),
   -B_ff * np.cos(psi_ff)
])

print("IK Optimal B [T]:", B_ff)
print("IK Optimal psi [deg]:", np.degrees(psi_ff))

# ============================================================
#   Common disturbance + control settings
# ============================================================

# Disturbance direction (fixed); we’ll sweep the magnitude
alpha_rel_true = np.pi / 6  # rad

# Time and control hyperparameters
dt      = 0.05
n_steps = 250               # longer horizon for better convergence

# Field magnitude limits
B_min = 0.005
B_max = 1.0               # slightly larger authority

# Closed-loop aggressiveness
alpha0      = 0.25          # outer scaling of Δu
du_max_norm = 1.0           # max |Δu| per iteration
lambda_reg  = 5e-5          # less damping → more aggressive

# Saturation-protection tolerance
err_tol = 5e-5

# ============================================================
#   Simulation function (uses disturbance-aware Jacobian)
# ============================================================

def run_simulation(use_control, u_init, theta_init,
                   params_model, params_true, p_des,
                   F_tip_true, alpha_rel_true,
                   dt=0.05, n_steps=250):
    """
    Run a simulation with or without closed-loop control under the
    given disturbance. Closed-loop uses disturbance-aware Jacobian
    based on online tip-force estimation.
    """

    # ---- Control gains ----
    if use_control:
        Kp = np.diag([0.06, 0.06])   # more aggressive
        Ki = np.diag([0.02, 0.02])
    else:
        Kp = np.diag([0.0, 0.0])
        Ki = np.diag([0.0, 0.0])

    # Histories
    u_hist   = np.zeros((n_steps+1, 2))
    B_hist   = np.zeros(n_steps+1)
    psi_hist = np.zeros(n_steps+1)
    p_hist   = np.zeros((n_steps+1, 2))
    e_hist   = np.zeros((n_steps+1, 2))

    u = u_init.copy()
    u_hist[0] = u

    e_int = np.zeros(2)

    # First equilibrium solve with disturbed true plant
    theta_guess_plant = theta_init.copy()
    p_tip, _, _, theta_eq_true, _ = forward_equilibrium_BxBy_true(
        u, theta_guess_plant, params_true,
        F_tip=F_tip_true, alpha_rel=alpha_rel_true
    )
    theta_guess_plant = theta_eq_true.copy()

    # Model side initial guess for Jacobian linearizations
    theta_guess_model = theta_guess_plant.copy()

    # Disturbance estimate (initial)
    F_hat_prev = 0.0
    alpha_hat_prev = 0.0

    # ---- Rollback / early-stop bookkeeping ----
    early_stop = False

    u_prev                 = u.copy()
    theta_guess_plant_prev = theta_guess_plant.copy()
    e_int_prev             = e_int.copy()
    p_prev = None
    e_prev = None
    B_prev = None
    psi_prev = None
    err_prev = None

    for k in range(n_steps + 1):

        # 1) True plant equilibrium with (true) disturbance
        p_tip, x_nodes, y_nodes, theta_eq_true, abs_angles_true = \
            forward_equilibrium_BxBy_true(
                u,
                theta_guess_plant,
                params_true,
                F_tip=F_tip_true,
                alpha_rel=alpha_rel_true,
                tol=1e-6, maxiter=200
            )
        theta_guess_plant = theta_eq_true.copy()
        p_hist[k] = p_tip

        # Current B, psi
        B_curr, psi_curr = Bvec_to_B_psi(u[0], u[1])
        B_hist[k]   = B_curr
        psi_hist[k] = psi_curr

        # 2) Task-space error
        e = p_des - p_tip
        e_hist[k] = e
        err_norm = np.linalg.norm(e)

        # 2.5) Saturation protection (only for closed loop)
        if use_control and (k > 0) and (err_prev is not None):
            if (B_curr >= 0.99 * B_max) and (err_norm > err_prev + err_tol):
                print(
                    f"[CL F={F_tip_true:.3f}] "
                    f"Error increased after saturation "
                    f"(prev {err_prev:.3e} -> now {err_norm:.3e}); "
                    f"rolling back one step and stopping."
                )

                u                   = u_prev.copy()
                theta_guess_plant   = theta_guess_plant_prev.copy()
                e_int               = e_int_prev.copy()

                p_hist[k]   = p_prev
                e_hist[k]   = e_prev
                B_hist[k]   = B_prev
                psi_hist[k] = psi_prev
                u_hist[k]   = u_prev

                for j in range(k+1, n_steps+1):
                    p_hist[j]   = p_prev
                    e_hist[j]   = e_prev
                    B_hist[j]   = B_prev
                    psi_hist[j] = psi_prev
                    u_hist[j]   = u_prev

                early_stop = True
                break

        # Update "previous" state
        u_prev                 = u.copy()
        theta_guess_plant_prev = theta_guess_plant.copy()
        e_int_prev             = e_int.copy()
        p_prev                 = p_tip.copy()
        e_prev                 = e.copy()
        B_prev                 = B_curr
        psi_prev               = psi_curr
        err_prev               = err_norm

        # 3) Open-loop: no control, just log and move on
        if (not use_control) or (k == n_steps):
            if k < n_steps:
                u_hist[k+1] = u
            continue

        # ----------------------------------------------------
        # 4) Closed-loop PI control with disturbance-aware J
        # ----------------------------------------------------
        # PI in task space
        e_int += e * dt
        v = Kp @ e + Ki @ e_int      # desired Δp

        # --- Disturbance estimation from tip position ---
        F_init_est = max(1e-4, F_hat_prev if F_hat_prev > 0 else 0.5 * F_tip_true)
        alpha_init_est = alpha_hat_prev

        F_hat, alpha_hat, _ = estimate_tip_force_from_tip(
            u=u,
            p_tip_meas=p_tip,
            theta_init=theta_eq_true,   # current equilibrium is a good init
            params_true=params_true,
            F_init=F_init_est,
            alpha_init=alpha_init_est,
            F_bounds=(0.0, 0.5),
            alpha_bounds=(-np.pi, np.pi),
            tol=1e-6,
            maxiter=50
        )
        F_hat_prev = F_hat
        alpha_hat_prev = alpha_hat

        # --- Disturbance-aware Jacobian on the true plant ---
        J, theta_guess_model = numerical_jacobian_u_BxBy_true_forced(
            u=u,
            theta_guess=theta_guess_model,
            params_true=params_true,
            F_tip_hat=F_hat,
            alpha_rel_hat=alpha_hat,
            dBx=1e-3,
            dBy=1e-3
        )
        if k == 0:
            print(f"J(0) (disturbance-aware) for F≈{F_hat:.3f}:\n", J)

        # Damped least-squares inverse
        JJt   = J @ J.T
        J_dls = J.T @ np.linalg.inv(JJt + (lambda_reg**2) * np.eye(2))

        alpha = alpha0 / (1.0 + err_norm / 0.05)
        du = J_dls @ v

        # Clamp Δu magnitude
        du_norm = np.linalg.norm(du)
        if du_norm > du_max_norm:
            du *= du_max_norm / (du_norm + 1e-9)

        u_raw = u + alpha * du

        # Enforce bounds on |B|
        B_mag_raw = np.linalg.norm(u_raw)
        u_sat = u_raw.copy()
        if B_mag_raw < B_min:
            u_sat *= B_min / (B_mag_raw + 1e-9)
        elif B_mag_raw > B_max:
            u_sat *= B_max / (B_mag_raw + 1e-9)

        u = u_sat
        u_hist[k+1] = u

    # ---- Final evaluation for this run ----
    p_tip_final, x_nodes_final, y_nodes_final, theta_final, abs_angles_final = \
        forward_equilibrium_BxBy_true(
            u,
            theta_guess_plant,
            params_true,
            F_tip=F_tip_true,
            alpha_rel=alpha_rel_true
        )

    B_final, psi_final = Bvec_to_B_psi(u[0], u[1])
    final_error_norm = np.linalg.norm(p_des - p_tip_final)

    label = "Closed-loop" if use_control else "Open-loop"
    print(f"\n=== {label} results (F_tip = {F_tip_true:.3f} N) ===")
    print("Final u = [Bx, By]:", u)
    print("Final B [T]:", B_final)
    print("Final psi [deg]:", np.degrees(psi_final))
    print("Final tip position [m]:", p_tip_final)
    print("Desired tip [m]:", p_des)
    print("Final tip error [m]:", final_error_norm)

    t = np.arange(n_steps+1) * dt

    results = dict(
        use_control=use_control,
        F_tip_true=F_tip_true,
        t=t,
        u_hist=u_hist,
        B_hist=B_hist,
        psi_hist=psi_hist,
        p_hist=p_hist,
        e_hist=e_hist,
        x_nodes_final=x_nodes_final,
        y_nodes_final=y_nodes_final,
        abs_angles_final=abs_angles_final,
        theta_final=theta_final,
        u_final=u,
        B_final=B_final,
        psi_final=psi_final,
        p_tip_final=p_tip_final,
        final_error_norm=final_error_norm,
        early_stop=early_stop,
        F_hat_last=F_hat_prev,
        alpha_hat_last=alpha_hat_prev
    )
    return results

# ============================================================
#   Disturbance sweep: multiple pairs of simulations
# ============================================================

# MATLAB-style 0:0.02:0.4
disturbance_magnitudes = [0, 0.025 , 0.05 ,0.075, 0.1]
#disturbance_magnitudes = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]

errors_open   = []
errors_closed = []

F_est_closed  = []
F_est_true    = []

results_open_last   = None
results_closed_last = None

for F_tip_true in disturbance_magnitudes:
    print("\n\n###############################")
    print(f"Running simulations for F_tip = {F_tip_true:.3f} N")
    print("###############################")

    # Initial guess for angles for this disturbance
    theta_init = theta0_hat * np.ones(N)

    res_open = run_simulation(
        use_control=False,
        u_init=u0,
        theta_init=theta_init,
        params_model=params_model,
        params_true=params_true,
        p_des=p_des,
        F_tip_true=F_tip_true,
        alpha_rel_true=alpha_rel_true,
        dt=dt,
        n_steps=n_steps
    )

    res_closed = run_simulation(
        use_control=True,
        u_init=u0,
        theta_init=theta_init,
        params_model=params_model,
        params_true=params_true,
        p_des=p_des,
        F_tip_true=F_tip_true,
        alpha_rel_true=alpha_rel_true,
        dt=dt,
        n_steps=n_steps
    )

    errors_open.append(res_open["final_error_norm"])
    errors_closed.append(res_closed["final_error_norm"])

    # Grab the last disturbance estimate from closed-loop
    F_est_closed.append(res_closed["F_hat_last"])
    F_est_true.append(F_tip_true)

    # Keep last pair for geometry visualization
    results_open_last   = res_open
    results_closed_last = res_closed

# Convert to arrays
disturbance_magnitudes = np.array(disturbance_magnitudes)
errors_open   = np.array(errors_open)
errors_closed = np.array(errors_closed)
F_est_closed  = np.array(F_est_closed)
F_est_true    = np.array(F_est_true)

# ============================================================
#   1) Final error vs disturbance magnitude
# ============================================================

plt.figure(figsize=(8, 4))
plt.plot(disturbance_magnitudes, errors_open,   'o--', label='Open-loop')
plt.plot(disturbance_magnitudes, errors_closed, 's-',  label='Closed-loop (dist.-aware)')
plt.xlabel('Tip disturbance magnitude F_tip [N]')
plt.ylabel('Final tip error norm [m]')
plt.title('Final tip error vs disturbance magnitude\nOpen vs Closed loop (disturbance-aware Jacobian)')
plt.grid(True)
plt.legend()
plt.tight_layout()

# ============================================================
#   2) Geometry comparison for final disturbance value
# ============================================================

if results_open_last is not None and results_closed_last is not None:
    x_open   = results_open_last["x_nodes_final"]
    y_open   = results_open_last["y_nodes_final"]
    x_closed = results_closed_last["x_nodes_final"]
    y_closed = results_closed_last["y_nodes_final"]

    abs_angles_open   = results_open_last["abs_angles_final"]
    abs_angles_closed = results_closed_last["abs_angles_final"]

    p_tip_open   = results_open_last["p_tip_final"]
    p_tip_closed = results_closed_last["p_tip_final"]

    F_last = results_open_last["F_tip_true"]

    plt.figure()
    ax = plt.gca()

    # Catheter backbones
    ax.plot(x_open,   y_open,   '--r', linewidth=2, label='Open-loop')
    ax.plot(x_closed, y_closed, '-b',  linewidth=2, label='Closed-loop')
    ax.plot(x_open,   y_open,   'ro', markersize=3)
    ax.plot(x_closed, y_closed, 'bo', markersize=3)

    # Desired and final tips
    ax.plot(p_des[0],       p_des[1],        'kx', markersize=8, mew=2, label='Desired tip')
    ax.plot(p_tip_open[0],  p_tip_open[1],   'rs', markersize=6,  label='Open-loop tip')
    ax.plot(p_tip_closed[0], p_tip_closed[1],'g^', markersize=6,  label='Closed-loop tip')

    # Disturbance directions (true)
    theta_tip_open   = abs_angles_open[-1]
    theta_tip_closed = abs_angles_closed[-1]

    psi_F_open   = theta_tip_open   + alpha_rel_true
    psi_F_closed = theta_tip_closed + alpha_rel_true

    Fhat_open_dir   = np.array([np.sin(psi_F_open),   -np.cos(psi_F_open)])
    Fhat_closed_dir = np.array([np.sin(psi_F_closed), -np.cos(psi_F_closed)])

    L_F = 0.005
    ax.quiver(
        p_tip_open[0], p_tip_open[1],
        L_F * Fhat_open_dir[0], L_F * Fhat_open_dir[1],
        angles='xy', scale_units='xy', scale=1,
        color='r', width=0.004, label='Disturbance (open)'
    )
    ax.quiver(
        p_tip_closed[0], p_tip_closed[1],
        L_F * Fhat_closed_dir[0], L_F * Fhat_closed_dir[1],
        angles='xy', scale_units='xy', scale=1,
        color='g', width=0.004, label='Disturbance (closed)'
    )

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(f'Final catheter pose at F_tip = {F_last:.3f} N')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()

# ============================================================
#   3) Force estimation accuracy (true vs estimated, closed-loop)
# ============================================================

plt.figure(figsize=(5, 4))
plt.plot(F_est_true, F_est_closed, 'o-', label='Estimated')
plt.plot(F_est_true, F_est_true, 'k--', label='Ideal')
plt.xlabel('True F_tip [N]')
plt.ylabel('Estimated F_tip [N]')
plt.title('Magnetics-based tip force estimation\n(from tip position, closed-loop)')
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()
