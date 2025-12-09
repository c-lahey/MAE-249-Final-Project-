import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from catheter_IK3 import inverse_kinematics_tip

# ============================================================
#   Physical parameters (model vs. true plant)
# ============================================================

# "Nominal" model parameters (used by controller / Jacobian)
E_nom = 1_250_000.0   # Pa
d_nom = 0.003         # m
I_nom = (np.pi * (d_nom/2)**4) / 4.0
l_nom = 0.004         # m
K_nom = E_nom * I_nom / l_nom

N  = 12               # number of links
B_nom = 0.015         # nominal B magnitude
mu_nom = 0.0005       # magnetic dipole strength
phi = np.zeros(N)     # axial polarization

params_model = dict(
    E=E_nom, d=d_nom, I=I_nom, l=l_nom, K=K_nom,
    N=N, B=B_nom, mu=mu_nom, phi=phi
)

# "True" plant parameters (mismatch for robustness test)
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
#   Energy, equilibrium, backbone geometry
# ============================================================

def catheter_energy(theta, psi, params):
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


def catheter_energy_true(theta, psi, params, M_tip=0.0):
    """
    True plant energy = model energy + tip torque disturbance.
    M_tip has units of N·m, applied about the tip angle.
    """
    theta = np.asarray(theta)

    E_nom_val = catheter_energy(theta, psi, params)

    abs_angles = np.cumsum(theta)
    theta_tip = abs_angles[-1]

    # Disturbance potential: - M * theta_tip
    PI_dist = -M_tip * theta_tip

    return E_nom_val + PI_dist


def solve_equilibrium(theta0, psi, params, tol=1e-6, maxiter=200):
    """Minimize nominal energy w.r.t. theta using BFGS."""
    fun = lambda th: catheter_energy(th, psi, params)
    res = minimize(
        fun,
        x0=np.asarray(theta0),
        method="BFGS",
        options=dict(gtol=tol, maxiter=maxiter, disp=False)
    )
    return res.x


def solve_equilibrium_true(theta0, psi, params, M_tip=0.0,
                           tol=1e-6, maxiter=200):
    """Minimize true plant energy (with tip moment disturbance)."""
    fun = lambda th: catheter_energy_true(th, psi, params, M_tip)
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
                                  M_tip=0.0, tol=1e-6, maxiter=200):
    """
    Forward map for given u on the *true* plant including tip moment.
    """
    Bx, By = u
    B_val, psi_val = Bvec_to_B_psi(Bx, By)

    p_local = params.copy()
    p_local["B"] = B_val

    theta_eq = solve_equilibrium_true(theta_guess, psi_val, p_local,
                                      M_tip=M_tip, tol=tol, maxiter=maxiter)
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
    u = [Bx, By], using the nominal (undisturbed) model.
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
#   Desired tip, IK feedforward field
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
Kp_CL = np.diag([0.1, 0.1])
Ki_CL = np.diag([0.1, 0.1])

dt      = 0.05
n_steps = 40

# Field magnitude limits
B_min = 0.005
B_max = 0.05

# Step-size / regularization
alpha0      = 0.1      # base step size
du_max_norm = 0.5      # max ||Δu|| per step (T)
lambda_reg  = 1e-4     # damping for Jacobian inverse

# ============================================================
#   Single-run simulation (open-loop or closed-loop)
# ============================================================

def run_simulation(use_control, u_init, theta_init,
                   params_model, params_true, p_des,
                   M_tip_true,
                   dt=0.05, n_steps=40):
    """
    If use_control is False → open-loop (u fixed).
    If use_control is True  → Jacobian PI closed-loop on tip position.
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
    p_tip, _, _, theta_eq_true, _ = forward_equilibrium_BxBy_true(
        u, theta_guess_plant, params_true, M_tip=M_tip_true
    )
    theta_guess_plant = theta_eq_true.copy()
    theta_guess_model = theta_guess_plant.copy()

    for k in range(n_steps + 1):

        # 1) True plant step with tip moment
        p_tip, x_nodes, y_nodes, theta_eq_true, abs_angles_true = \
            forward_equilibrium_BxBy_true(
                u,
                theta_guess_plant,
                params_true,
                M_tip=M_tip_true,
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
            # Open-loop: keep u constant, just log
            if k < n_steps:
                u_hist[k+1] = u
            continue

        # 3) PI control in task space
        e_int += e * dt
        v = Kp @ e + Ki @ e_int   # desired Δp

        # 4) Nominal Jacobian (no disturbance in model)
        J, theta_guess_model = numerical_jacobian_u_BxBy(
            u, theta_guess_model, params_model,
            dBx=1e-3, dBy=1e-3
        )
        if k == 0:
            label = "Closed-loop" if use_control else "Open-loop"
            print(f"J(0) for {label}, M_tip = {M_tip_true:.2e} N·m:\n", J)

        # 5) Damped least-squares inverse + clamped Δu
        JJt   = J @ J.T
        J_dls = J.T @ np.linalg.inv(JJt + (lambda_reg**2) * np.eye(2))

        err_norm = np.linalg.norm(e)
        alpha = alpha0 / (1.0 + err_norm / 0.05)

        du = J_dls @ v

        du_norm = np.linalg.norm(du)
        if du_norm > du_max_norm:
            du *= du_max_norm / (du_norm + 1e-9)

        u = u + alpha * du

        # Enforce bounds on |B|
        B_mag = np.linalg.norm(u)
        if B_mag < B_min:
            u *= B_min / (B_mag + 1e-9)
        elif B_mag > B_max:
            u *= B_max / (B_mag + 1e-9)

        u_hist[k+1] = u

    # Final evaluation
    p_tip_final, x_nodes_final, y_nodes_final, theta_final, abs_angles_final = \
        forward_equilibrium_BxBy_true(
            u,
            theta_guess_plant,
            params_true,
            M_tip=M_tip_true
        )

    B_final, psi_final = Bvec_to_B_psi(u[0], u[1])
    final_err_norm = np.linalg.norm(p_des - p_tip_final)

    label = "Closed-loop" if use_control else "Open-loop"
    print(f"\n=== {label} results (M_tip = {M_tip_true:.2e} N·m) ===")
    print("Final u = [Bx, By]:", u)
    print("Final B [T]:", B_final)
    print("Final psi [deg]:", np.degrees(psi_final))
    print("Final tip position [m]:", p_tip_final)
    print("Desired tip [m]:", p_des)
    print("Final tip error [m]:", final_err_norm)

    t = np.arange(n_steps+1) * dt

    results = dict(
        use_control=use_control,
        M_tip_true=M_tip_true,
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
#   Disturbance sweep: multiple pairs of simulations
# ============================================================

# Define a list of tip moments to test (negative bends in one direction)

M_tip_list = np.array([0.0,
                       -5e-6,
                       -10e-6,
                       -15e-6,
                       -20e-6,
                       -25e-6,
                       -30e-6,
                       -35e-6,
                       -40e-6,
                       #-50e-6,])
])
'''
M_tip_list = np.array([0.0,
                       -5e-6,
                       -10e-6,
                       -15e-6,
                       -20e-6,
                       -25e-6,
                       -30e-6,
                       -35e-6,
                       -40e-6,
                       -50e-6,
                       -60e-6,
                       -70e-6,
                       -80e-6,
                       -90e-6,
                       -100e-6])
                       '''

errors_open   = []
errors_closed = []

results_open_last   = None
results_closed_last = None

for M_tip_true in M_tip_list:
    print("\n\n###############################")
    print(f"Running simulations for M_tip = {M_tip_true:.2e} N·m")
    print("###############################")

    theta_init = theta0_hat * np.ones(N)

    res_open = run_simulation(
        use_control=False,
        u_init=u0,
        theta_init=theta_init,
        params_model=params_model,
        params_true=params_true,
        p_des=p_des,
        M_tip_true=M_tip_true,
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
        M_tip_true=M_tip_true,
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
#   1) Final error vs disturbance magnitude
# ============================================================

plt.figure(figsize=(8, 4))
plt.plot(np.abs(M_tip_list)*1e6, errors_open,   'o--', label='Open-loop')
plt.plot(np.abs(M_tip_list)*1e6, errors_closed, 's-',  label='Closed-loop')
plt.xlabel('Tip disturbance moment |M_tip| [µN·m]')
plt.ylabel('Final tip error norm [m]')
plt.title('Final tip error vs disturbance moment\nOpen vs Closed loop')
plt.grid(True)
plt.legend()
plt.tight_layout()

# ============================================================
#   2) Geometry comparison for largest |M_tip|
# ============================================================

if results_open_last is not None and results_closed_last is not None:
    x_open   = results_open_last["x_nodes_final"]
    y_open   = results_open_last["y_nodes_final"]
    x_closed = results_closed_last["x_nodes_final"]
    y_closed = results_closed_last["y_nodes_final"]

    p_tip_open   = results_open_last["p_tip_final"]
    p_tip_closed = results_closed_last["p_tip_final"]
    M_last       = results_open_last["M_tip_true"]

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
    ax.set_title(f'Final catheter pose at M_tip = {M_last:.2e} N·m')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()

plt.show()
