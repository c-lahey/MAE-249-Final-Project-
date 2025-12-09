import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from catheter_IK3 import inverse_kinematics_tip

# ============================================================
#   Physical parameters (nominal model)
# ============================================================

E_nom = 1_250_000.0   # Pa
d_nom = 0.003         # m
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

# ============================================================
#   Energy, equilibrium, backbone geometry
# ============================================================

def catheter_energy(theta, psi, params):
    """
    Nominal magnetic + bending energy.
    """
    theta = np.asarray(theta)
    K   = params["K"]
    B   = params["B"]
    mu  = params["mu"]
    phi = params["phi"]

    PI_bend = 0.5 * K * np.sum(theta**2)

    abs_angles = np.cumsum(theta)
    arg = -(phi + abs_angles) + psi
    PI_mag = -mu * B * np.sum(np.cos(arg))

    return PI_bend + PI_mag


def solve_equilibrium(theta0, psi, params, tol=1e-6, maxiter=200):
    """Minimize energy w.r.t. theta using BFGS."""
    fun = lambda th: catheter_energy(th, psi, params)
    res = minimize(
        fun,
        x0=np.asarray(theta0),
        method="BFGS",
        options=dict(gtol=tol, maxiter=maxiter, disp=False)
    )
    return res.x


def backbone_xy(abs_angles, l):
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
#   Forward maps
# ============================================================

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
    Forward map for a given field u = [Bx, By] on the given params.
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

# ============================================================
#   Numerical Jacobian wrt u = [Bx, By] (nominal model)
# ============================================================

def numerical_jacobian_u_BxBy(u, theta_guess, params,
                              dBx=1e-4, dBy=1e-4):
    """
    Compute J = ∂p/∂u numerically using the nominal model.
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
#   Desired tip & IK feedforward field (nominal)
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

print("IK feedforward B [T]:", B_ff)
print("IK feedforward psi [deg]:", np.degrees(psi_ff))

# ============================================================
#   Control / simulation parameters
# ============================================================

# PI gains in task space (for closed loop)
Kp_CL = np.diag([0.01, 0.01])
Ki_CL = np.diag([0.01, 0.01])

dt      = 0.05
n_steps = 250

# Field magnitude limits
B_min = 0.0005
B_max = 0.75

# Step-size / regularization
alpha0      = 0.1      # base step size
du_max_norm = 0.1      # max ||Δu|| per step (T)
lambda_reg  = 1e-4     # damping for Jacobian inverse

# ============================================================
#   Single-run simulation (open- or closed-loop)
# ============================================================

def run_simulation(use_control, u_init, theta_init,
                   params_model, params_true, p_des,
                   dt=0.05, n_steps=100):
    """
    If use_control is False → open-loop (u fixed).
    If use_control is True  → Jacobian PI closed-loop on tip position.

    params_true encodes the "disturbed" plant (e.g. stiffness mismatch).
    """

    # Gains
    if use_control:
        Kp = Kp_CL
        Ki = Ki_CL
    else:
        Kp = np.zeros((2, 2))
        Ki = np.zeros((2, 2))

    # Histories
    u_hist   = np.zeros((n_steps+1, 2))
    B_hist   = np.zeros(n_steps+1)
    psi_hist = np.zeros(n_steps+1)
    p_hist   = np.zeros((n_steps+1, 2))
    e_hist   = np.zeros((n_steps+1, 2))

    u = u_init.copy()
    u_hist[0] = u
    e_int = np.zeros(2)

    # Initial guesses
    theta_guess_plant = theta_init.copy()
    p_tip, _, _, theta_eq_true, _ = forward_equilibrium_BxBy(
        u, theta_guess_plant, params_true
    )
    theta_guess_plant = theta_eq_true.copy()
    theta_guess_model = theta_guess_plant.copy()

    for k in range(n_steps + 1):

        # 1) True plant with parameter mismatch
        p_tip, x_nodes, y_nodes, theta_eq_true, abs_angles_true = \
            forward_equilibrium_BxBy(
                u,
                theta_guess_plant,
                params_true,
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

        if (not use_control) or (k == n_steps):
            # Open-loop: just hold u constant (or on final step)
            if k < n_steps:
                u_hist[k+1] = u
            continue

        # 3) PI control in task space (with simple anti-windup)
        #    -> We only update e_int if we are *not* saturated.
        v = Kp @ e + Ki @ e_int   # desired tip correction Δp from current integral

        # 4) Nominal Jacobian (model uses nominal params)
        J, theta_guess_model = numerical_jacobian_u_BxBy(
            u, theta_guess_model, params_model,
            dBx=1e-3, dBy=1e-3
        )
        if k == 0:
            label = "Closed-loop" if use_control else "Open-loop"
            print(f"J(0) for {label}:\n", J)

        # 5) Damped least-squares inverse
        JJt   = J @ J.T
        J_dls = J.T @ np.linalg.inv(JJt + (lambda_reg**2) * np.eye(2))

        err_norm = np.linalg.norm(e)
        alpha = alpha0 / (1.0 + err_norm / 0.05)

        du = J_dls @ v

        # Clamp Δu magnitude
        du_norm = np.linalg.norm(du)
        if du_norm > du_max_norm:
            du *= du_max_norm / (du_norm + 1e-9)

        # Tentative update
        u_new = u + alpha * du

        # 6) Saturation in |B| with anti-windup on e_int
        B_mag = np.linalg.norm(u_new)
        saturated = False

        if B_mag < B_min:
            u_new *= B_min / (B_mag + 1e-9)
            saturated = True
        elif B_mag > B_max:
            u_new *= B_max / (B_mag + 1e-9)
            saturated = True

        # Anti-windup: only integrate error when not saturated
        if not saturated:
            e_int += e * dt

        # Commit control input and log
        u = u_new
        u_hist[k+1] = u


    # Final evaluation
    p_tip_final, x_nodes_final, y_nodes_final, theta_final, abs_angles_final = \
        forward_equilibrium_BxBy(
            u,
            theta_guess_plant,
            params_true
        )

    B_final, psi_final = Bvec_to_B_psi(u[0], u[1])
    final_err_norm = np.linalg.norm(p_des - p_tip_final)

    label = "Closed-loop" if use_control else "Open-loop"
    print(f"\n=== {label} results (E_true/E_nom = {params_true['E']/E_nom:.2f}) ===")
    print("Final u = [Bx, By]:", u)
    print("Final B [T]:", B_final)
    print("Final psi [deg]:", np.degrees(psi_final))
    print("Final tip position [m]:", p_tip_final)
    print("Desired tip [m]:", p_des)
    print("Final tip error [m]:", final_err_norm)

    t = np.arange(n_steps+1) * dt

    results = dict(
        use_control=use_control,
        E_true=params_true["E"],
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
        final_error_norm=final_err_norm
    )
    return results

# ============================================================
#   Parameter-mismatch sweep: E_true/E_nom
# ============================================================

# Relative stiffness factors to test (true plant)
stiffness_factors = np.array([0.7, 0.75, 0.8, 0.9, 1.0])

errors_open   = []
errors_closed = []

results_open_last   = None
results_closed_last = None

for scale in stiffness_factors:
    print("\n\n###############################")
    print(f"Running simulations for stiffness scale = {scale:.2f}")
    print("###############################")

    # Build true-plant params for this scale
    E_true = scale * E_nom
    K_true = scale * K_nom   # linear in E
    params_true = dict(
        E=E_true, d=d_nom, I=I_nom, l=l_nom, K=K_true,
        N=N, B=B_nom, mu=mu_nom, phi=phi
    )

    theta_init = theta0_hat * np.ones(N)

    res_open = run_simulation(
        use_control=False,
        u_init=u0,
        theta_init=theta_init,
        params_model=params_model,
        params_true=params_true,
        p_des=p_des,
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
        dt=dt,
        n_steps=n_steps
    )

    errors_open.append(res_open["final_error_norm"])
    errors_closed.append(res_closed["final_error_norm"])

    results_open_last   = res_open
    results_closed_last = res_closed

errors_open   = np.array(errors_open)
errors_closed = np.array(errors_closed)

# ============================================================
#   1) Final error vs stiffness mismatch
# ============================================================

plt.figure(figsize=(8, 4))
plt.plot(stiffness_factors, errors_open,   'o--', label='Open-loop')
plt.plot(stiffness_factors, errors_closed, 's-',  label='Closed-loop')
plt.xlabel('Relative stiffness E_true / E_nom [-]')
plt.ylabel('Final tip error norm [m]')
plt.title('Final tip error vs stiffness mismatch\nOpen vs Closed loop')
plt.grid(True)
plt.legend()
plt.tight_layout()

# ============================================================
#   2) Geometry comparison for largest mismatch
# ============================================================

if results_open_last is not None and results_closed_last is not None:
    x_open   = results_open_last["x_nodes_final"]
    y_open   = results_open_last["y_nodes_final"]
    x_closed = results_closed_last["x_nodes_final"]
    y_closed = results_closed_last["y_nodes_final"]

    p_tip_open   = results_open_last["p_tip_final"]
    p_tip_closed = results_closed_last["p_tip_final"]
    E_last       = results_open_last["E_true"]

    plt.figure()
    ax = plt.gca()

    ax.plot(x_open,   y_open,   '--r', linewidth=2, label='Open-loop')
    ax.plot(x_closed, y_closed, '-b',  linewidth=2, label='Closed-loop')
    ax.plot(x_open,   y_open,   'ro', markersize=3)
    ax.plot(x_closed, y_closed, 'bo', markersize=3)

    ax.plot(p_des[0],        p_des[1],        'kx', markersize=8, mew=2, label='Desired tip')
    ax.plot(p_tip_open[0],   p_tip_open[1],   'rs', markersize=6,       label='Open-loop tip')
    ax.plot(p_tip_closed[0], p_tip_closed[1], 'g^', markersize=6,       label='Closed-loop tip')

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(f'Final catheter pose at E_true/E_nom = {E_last/E_nom:.2f}')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()

plt.show()
