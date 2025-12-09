import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from catheter_IK3 import inverse_kinematics_tip

# ============================================================
#   Physical parameters (model vs. true plant)
# ============================================================

# "Nominal" model parameters (used by controller / Jacobian)
E_nom = 1250000     # Pa
d_nom = 0.004       # m
I_nom = (np.pi * (d_nom/2)**4) / 4.0
l_nom = 0.004       # m
K_nom = E_nom * I_nom / l_nom

N  = 12             # number of links
B_nom = 0.015       # nominal B magnitude
mu_nom = 0.005     # magnetic dipole strength
phi = np.zeros(N)   # axial polarization

params_model = dict(
    E=E_nom, d=d_nom, I=I_nom, l=l_nom, K=K_nom,
    N=N, B=B_nom, mu=mu_nom, phi=phi
)

# "True" plant parameters (mismatch for robustness test)
E_true = 1.2*E_nom
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
    B = np.sqrt(Bx**2 + By**2)
    if B < B_min:
        B = B_min
    # Bx = B sin psi, By = -B cos psi
    psi = np.arctan2(Bx, -By)
    return B, psi


def forward_equilibrium_BxBy(u, theta_guess, params,
                             tol=1e-6, maxiter=200):
    """
    Forward map for a given field u = [Bx, By] (nominal model).
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
#   Numerical Jacobian wrt u = [Bx, By]
# ============================================================

def numerical_jacobian_u_BxBy(u, theta_guess, params,
                              dBx=1e-4, dBy=1e-4):
    """
    Compute J = ∂p/∂u numerically, where p = [x_tip, y_tip],
    u = [Bx, By].
    """
    u = np.asarray(u, dtype=float)

    # Base point
    p0, _, _, theta0, _ = forward_equilibrium_BxBy(u, theta_guess, params)
    theta_base = theta0.copy()

    # Perturb Bx
    uBx = u.copy()
    uBx[0] += dBx
    pBx, _, _, thetaBx, _ = forward_equilibrium_BxBy(uBx, theta_base, params)
    dp_dBx = (pBx - p0) / dBx

    # Perturb By
    uBy = u.copy()
    uBy[1] += dBy
    pBy, _, _, thetaBy, _ = forward_equilibrium_BxBy(uBy, theta_base, params)
    dp_dBy = (pBy - p0) / dBy

    J = np.column_stack((dp_dBx, dp_dBy))   # shape (2,2)

    return J, thetaBy

# ============================================================
#   Desired tip, feedforward field via IK (shared)
# ============================================================

x_des = 0.0322
y_des = -0.0319
p_des = np.array([x_des, y_des])

theta0_hat = np.arctan2(y_des, x_des)
theta_init = theta0_hat * np.ones(N)

field_opt, _, _, _, _, _ = inverse_kinematics_tip(
    x_des, y_des,
    theta_init, params_model,
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
n_steps = 250

# Field magnitude limits (only used when control is ON)
B_min = 0.005 
B_max = 1

alpha0      = 0.1
du_max_norm = 0.5
lambda_reg  = 1e-4

# Saturation-protection tolerance
err_tol = 2e-5

# ============================================================
#   Simulation function (runs either open- or closed-loop)
# ============================================================

def run_simulation(use_control, u_init, theta_init,
                   params_model, params_true, p_des,
                   F_tip_true, alpha_rel_true,
                   dt=0.05, n_steps=100):
    """
    Run a simulation with or without closed-loop control under the
    given disturbance. Returns a dict of results.
    """

    # ---- Control gains ----
    if use_control:
        Kp = np.diag([0.025, 0.025])
        Ki = np.diag([0.01, 0.01])
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

    # Model side initial guess
    theta_guess_model = theta_guess_plant.copy()

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

        # 1) True plant equilibrium with disturbance
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
                    f"[{'CL' if use_control else 'OL'} F={F_tip_true:.3f}] "
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

        # 4) Closed-loop PI control
        e_int += e * dt
        v = Kp @ e + Ki @ e_int      # desired Δp

        J, theta_guess_model = numerical_jacobian_u_BxBy(
            u, theta_guess_model, params_model,
            dBx=1e-3, dBy=1e-3
        )
        if k == 0:
            print(f"J(0) for F={F_tip_true:.3f}:\n", J)

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
        t=t,
        u_hist=u_hist,
        B_hist=B_hist,
        psi_hist=psi_hist,
        p_hist=p_hist,
        e_hist=e_hist,
        x_nodes_final=x_nodes_final,
        y_nodes_final=y_nodes_final,
        abs_angles_final=abs_angles_final,
        u_final=u,
        B_final=B_final,
        psi_final=psi_final,
        p_tip_final=p_tip_final,
        final_error_norm=final_error_norm,
        F_tip_true=F_tip_true,
        early_stop=early_stop
    )
    return results

# ============================================================
#   Disturbance sweep: multiple pairs of simulations
# ============================================================

# EDIT THIS LIST for your report:
disturbance_magnitudes = [0, 0.025, 0.05, 0.075, 0.1 ]
#disturbance_magnitudes = [0.02 * i for i in range(0, 21)]


errors_open   = []
errors_closed = []

results_open_last   = None
results_closed_last = None

for F_tip_true in disturbance_magnitudes:
    print("\n\n###############################")
    print(f"Running simulations for F_tip = {F_tip_true:.3f} N")
    print("###############################")

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

    # Keep last pair for geometry visualization
    results_open_last   = res_open
    results_closed_last = res_closed

# ============================================================
#   1) Final error vs disturbance magnitude
# ============================================================

disturbance_magnitudes = np.array(disturbance_magnitudes)
errors_open   = np.array(errors_open)
errors_closed = np.array(errors_closed)

plt.figure()
plt.plot(disturbance_magnitudes, errors_open,   'o--', label='Open-loop')
plt.plot(disturbance_magnitudes, errors_closed, 's-',  label='Closed-loop')
plt.xlabel('Tip disturbance magnitude F_tip [N]')
plt.ylabel('Final tip error norm [m]')
plt.title('Final tip error vs disturbance magnitude\nOpen vs Closed loop')
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

    # Disturbance directions
    theta_tip_open   = abs_angles_open[-1]
    theta_tip_closed = abs_angles_closed[-1]

    psi_F_open   = theta_tip_open   + alpha_rel_true
    psi_F_closed = theta_tip_closed + alpha_rel_true

    Fhat_open   = np.array([np.sin(psi_F_open),   -np.cos(psi_F_open)])
    Fhat_closed = np.array([np.sin(psi_F_closed), -np.cos(psi_F_closed)])

    L_F = 0.005
    ax.quiver(
        p_tip_open[0], p_tip_open[1],
        L_F * Fhat_open[0], L_F * Fhat_open[1],
        angles='xy', scale_units='xy', scale=1,
        color='r', width=0.004, label='Disturbance (open)'
    )
    ax.quiver(
        p_tip_closed[0], p_tip_closed[1],
        L_F * Fhat_closed[0], L_F * Fhat_closed[1],
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

plt.show()
