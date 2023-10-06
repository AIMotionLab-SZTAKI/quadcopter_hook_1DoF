from quadcopter_hook_onedof.ctrl.casadi_model import eval_model
import casadi as ca


def compute_delta(eq_no=1):
    opti = ca.Opti()
    drone_mass = opti.parameter()
    opti.set_value(drone_mass, 0.605)
    payload_mass = opti.parameter()
    opti.set_value(payload_mass, 0.07)
    g = opti.parameter()
    opti.set_value(g, 9.81)
    L = opti.parameter()
    opti.set_value(L, 0.4)
    J = ca.diag(ca.DM([1.5e-3, 1.5e-3, 2.66e-3]))
    invJ = ca.inv(J)  # Element-wise inverse is okay here
    e3 = ca.DM([[0.], [0.], [1.]])

    param_num = [drone_mass, payload_mass, g, L, J[0, 0], J[1, 1], J[2, 2]]

    v = opti.variable(3)
    lam = opti.variable(3)
    dlam = opti.variable(3)
    F = opti.variable()
    tau = opti.variable(3)
    alpha = opti.variable()
    dalpha = opti.variable()

    setpoint = [0., 0., 0., lam[0], lam[1], lam[2], alpha, v[0], v[1], v[2], dlam[0], dlam[1], dlam[2], dalpha,
                F + (drone_mass + payload_mass) * g, tau[0], tau[1], tau[2]]

    model = eval_model()
    R = model['fR'](*(setpoint + param_num))
    Q = model['fQ'](*(setpoint + param_num))
    invQ = model['finvQ'](*(setpoint + param_num))
    dQ = model['fdQ'](*(setpoint + param_num))
    f_rhs = model['f'](*(setpoint + param_num))

    om = Q @ dlam
    u = ca.vertcat(F + (drone_mass + payload_mass) * g, tau)

    eq11 = -g * e3 + u[0] / (drone_mass + payload_mass) * R @ e3
    eq12 = invJ @ (u[1:] - ca.cross(om, J @ om))

    Q_ext = ca.MX(7, 7)
    Q_ext[0, 0] = 1
    Q_ext[1, 1] = 1
    Q_ext[2, 2] = 1
    Q_ext[6, 6] = 1
    Q_ext[3:6, 3:6] = Q

    eq2 = Q_ext @ f_rhs + ca.vertcat(ca.MX(3, 1), dQ @ invQ @ om, 0)
    eq21 = eq2[0:3]
    eq22 = eq2[3:6]
    delta1 = (drone_mass + payload_mass) * (eq21 - eq11)
    delta2 = J @ (eq22 - eq12)

    opti.subject_to([opti.bounded(-0.7, lam, 0.7), opti.bounded(-0.7, dlam, 0.7), opti.bounded(-0.5, alpha, 0.5),
                     opti.bounded(-0.5, dalpha, 0.5), opti.bounded(-0.7, F, 0.7), opti.bounded(-0.05, tau, 0.05),
                     opti.bounded(-1, v, 1)])
    if eq_no == 1:
        obj = -delta1[0]**2 - delta1[1]**2 - delta1[2]**2
    elif eq_no == 2:
        obj = -delta2[0]**2 - delta2[1]**2 - delta2[2]**2
    else:
        print('eq_no has to be 1 or 2 (1 for \delta_r, 2 for \delta_R)')
        return
    opti.minimize(obj)
    opti.set_initial(v, -1)
    opti.set_initial(lam, -0.7)
    opti.set_initial(dlam, 0.7)
    opti.set_initial(alpha, -0.5)
    opti.set_initial(dalpha, 0.5)
    opti.set_initial(F, -0.7)
    opti.set_initial(tau, 0.05)
    p_opts = {"expand": False}
    s_opts = {"max_iter": 100}
    opti.solver("ipopt", p_opts, s_opts)
    sol = opti.solve()
    print(f"Lambda: {sol.value(lam)}")
    print(f"Lambda_dot: {sol.value(dlam)}")
    print(f"Alpha: {sol.value(alpha)}")
    print(f"Alpha_dot: {sol.value(dalpha)}")
    print(f"Thrust: {sol.value(F)}")
    print(f"Torque: {sol.value(tau)}")
    print(f"Solution: {sol.value(obj)}")


if __name__ == '__main__':
    compute_delta(eq_no=1)
    compute_delta(eq_no=2)
