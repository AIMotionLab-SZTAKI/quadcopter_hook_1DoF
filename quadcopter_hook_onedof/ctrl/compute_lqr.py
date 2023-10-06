import numpy as np
import scipy.linalg
import sympy as sp
import scipy as si
import control
import inspect


def hat(a):
    mat = sp.Matrix([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return mat


def linearize_sys():
    m, mL, g, L, J, t = sp.symbols('m mL g L J t')
    x, z, phi, phiL, F, tau = sp.Function('x')(t), sp.Function('z')(t), sp.Function('phi')(t), sp.Function('phiL')(t), \
                              sp.Function('F')(t), sp.Function('tau')(t)
    dx, dz, dphi, dphiL = sp.diff(x, t), sp.diff(z, t), sp.diff(phi, t), sp.diff(phiL, t)
    xi = sp.Matrix([x, dx, z, dz, phi, dphi, phiL, dphiL])
    u = sp.Matrix([F, tau])
    f = sp.Matrix([dx, F*sp.sin(phi)/(m+mL), dz,
                   -g + F * sp.cos(phi) / (m + mL),
                   dphi, tau/J, dphiL, F*sp.sin(phi-phiL)/m/L])
    xi0 = sp.Matrix([0, 0, 0, 0, 0, 0, 0, 0])
    u0 = sp.Matrix([(m+mL)*g, 0])
    A = sp.diff(f, xi)
    B = sp.diff(f, u)
    A = A[:, 0, :, 0]
    B = B[:, 0, :, 0]
    sym_arr = sp.Matrix.vstack(xi, u)
    num_arr = sp.Matrix.vstack(xi0, u0)
    A = sp.Matrix([[A[i, j].subs([(var_sym, var_num) for var_sym, var_num in zip(sym_arr, num_arr)])
                    for i in range(8)] for j in range(8)])
    B = sp.Matrix([[B[i, j].subs([(var_sym, var_num) for var_sym, var_num in zip(sym_arr, num_arr)])
                    for i in range(2)] for j in range(8)])
    # print(sp.latex(A))
    # print(sp.latex(B))
    param_sym = [m, mL, g, L, J, sp.Derivative(0, t)]
    param_num = [0.605, 0.02, 9.81, 0.4, 1.5e-3, 0]
    # param_num = [0.605, 0.01, 9.81, 0.4, 1.5e-3, 0]
    A = sp.Matrix([[A[i, j].subs([(var_sym, var_num) for var_sym, var_num in zip(param_sym, param_num)])
                    for i in range(8)] for j in range(8)])
    B = sp.Matrix([[B[i, j].subs([(var_sym, var_num) for var_sym, var_num in zip(param_sym, param_num)])
                    for i in range(8)] for j in range(2)])
    A = np.array(A).T
    B = np.array(B).T
    expM = np.vstack((np.hstack((A, B)), np.zeros((2, 10))))*0.02
    blkmtx = si.linalg.expm(expM)
    A_d = blkmtx[0:8, 0:8]
    B_d = blkmtx[0:8, 8:]
    # Q = np.diag([1, 1, 1, 1, 1, 1, 1, 1])
    # R = 1e-2*np.eye(2)
    Q = np.diag([4, 2, 4, 2, 10, 2, 0.05, 0.05])
    R = 0.1*np.eye(2)
    K, S, E = control.dlqr(A_d, B_d, Q, R, method='scipy')
    return K


def linearize_full_sys():
    mb, ml, g, L, Jx, Jy, Jz, t = sp.symbols('m mL g L Jx Jy Jz t')
    # mb = 0.605
    # ml = 0.01
    # g = 9.81
    # L = 0.4
    # Jx, Jy, Jz = 1.5e-3, 1.5e-3, 2.66e-3
    # t = sp.symbols('t')
    J = sp.diag(Jx, Jy, Jz)
    phi, theta, psi, rx, ry, rz, alpha, F, taux, tauy, tauz = sp.Function('phi')(t), sp.Function('theta')(t), sp.Function('psi')(t), \
                                                 sp.Function('rx')(t), sp.Function('ry')(t), sp.Function('rz')(t), \
                                                 sp.Function('alpha')(t), sp.Function('F')(t), sp.Function('taux')(t), \
                                                              sp.Function('tauy')(t), sp.Function('tauz')(t)
    r = sp.Matrix([rx, ry, rz])
    q = sp.Matrix([rx, ry, rz, phi, theta, psi, alpha])
    dq = sp.diff(q, t)

    W = sp.Matrix([[sp.cos(theta)*sp.cos(psi), -sp.sin(psi), 0], [sp.cos(theta)*sp.sin(psi), sp.cos(psi), 0],
                   [-sp.sin(theta), 0, 1]])

    R = sp.Matrix([[sp.cos(psi), -sp.sin(psi), 0], [sp.sin(psi), sp.cos(psi), 0], [0, 0, 1]]) * \
        sp.Matrix([[sp.cos(theta), 0, sp.sin(theta)], [0, 1, 0], [-sp.sin(theta), 0, sp.cos(theta)]]) * \
        sp.Matrix([[1, 0, 0], [0, sp.cos(phi), -sp.sin(phi)], [0, sp.sin(phi), sp.cos(phi)]])

    # Q = R.T * W
    Q = sp.Matrix([[1, 0, -sp.sin(theta)], [0, sp.cos(phi), sp.sin(phi)*sp.cos(theta)],
                   [0, -sp.sin(phi), sp.cos(phi)*sp.cos(theta)]])
    invQ = sp.Matrix([[1, sp.sin(phi)*sp.tan(theta), sp.cos(phi)*sp.tan(theta)], [0, sp.cos(phi), -sp.sin(phi)],
                      [0, sp.sin(phi)/sp.cos(theta), sp.cos(phi)/sp.cos(theta)]])
    dinvQ = invQ.diff(t)

    Jp = sp.Matrix([-L * sp.cos(alpha), 0, L * sp.sin(alpha)])
    peb = sp.Matrix([-L * sp.sin(alpha), 0, -L * sp.cos(alpha)])

    H = sp.zeros(7)
    H[0:3, 0:3] = (mb + ml) * sp.eye(3)
    H[3:6, 3:6] = ml * W.T * hat(R * peb).T * hat(R * peb) * W + Q.T * J * Q
    H[6, 6] = ml * L ** 2
    H[0:3, 3:6] = -ml * hat(R * peb) * W
    H[3:6, 0:3] = H[0:3, 3:6].T
    H[0:3, 6] = ml * R * Jp
    H[6, 0:3] = H[0:3, 6].T
    H[3:6, 6] = -ml * W.T * hat(R * peb).T * R * Jp
    H[6, 3:6] = H[3:6, 6].T

    U = mb * g * rz + ml * g * (r + R * peb)[2]

    C = sp.zeros(7)
    for i, j in zip(range(7), range(7)):
        C[i, j] = sum([0.5 * (H[i, j].diff(q[k]) + H[i, k].diff(q[j])
                              + H[j, k].diff(q[i])) * q[k].diff(t) for k in range(7)])

    N = sp.Matrix([[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
                   [0, 0, 0, 0]])
    Rb = sp.eye(7)
    Rb[0:3, 0:3] = R
    Rb[3:6, 3:6] = Q.T
    Xi = Rb * N  # u = Xi * u_quad

    u = sp.Matrix([F, taux, tauy, tauz])

    xi = sp.Matrix([q, dq])
    xi0 = sp.zeros(14, 1)
    u0 = sp.Matrix([(mb + ml) * g, 0, 0, 0])
    sym_arr = sp.Matrix([xi, u, mb, ml, g, L, Jx, Jy, Jz, sp.Derivative(0, t)])
    num_arr = sp.Matrix([xi0, u0, 0.605, 0.01, 9.81, 0.4, 1.5e-3, 1.5e-3, 2.66e-3, 0])

    dXidq = (Xi * sp.Matrix([F, taux, tauy, tauz])).diff(q)[:, 0, :, 0]

    dCdq = (C * dq).diff(q)[:, 0, :, 0]

    G = U.diff(q)
    G_mat = sp.diff(G, q)
    G_mat = G_mat[:, 0, :, 0]

    # H = sp.simplify(H)
    # print("simplify phase 1 done")
    # H = sp.trigsimp(H)
    # print("simplification done")
    # H_lam = inspect.getsource(sp.lambdify(sym_arr, H))
    # print(H_lam)
    # invH_lam = inspect.getsource(sp.lambdify(sym_arr, H.inv()))
    # print(invH_lam)
    # C_lam = inspect.getsource(sp.lambdify(sym_arr, C))
    # Xi_lam = inspect.getsource(sp.lambdify(sym_arr, Xi))
    # dXidq_lam = inspect.getsource(sp.lambdify(sym_arr, dXidq))
    # G_lam = inspect.getsource(sp.lambdify(sym_arr, G_mat))
    # R_lam = inspect.getsource(sp.lambdify(sym_arr, R))
    # Q_lam = inspect.getsource(sp.lambdify(sym_arr, Q))
    # invQ_lam = inspect.getsource(sp.lambdify(sym_arr, invQ))
    # dinvQ_lam = inspect.getsource(sp.lambdify(sym_arr, dinvQ))
    # g_lam = inspect.getsource(sp.lambdify(sym_arr, G))


    H_lin = sp.Matrix([[H[i, j].subs([(var_sym, var_num) for var_sym, var_num in zip(sym_arr, num_arr)])
                        for i in range(7)] for j in range(7)])
    H_lin = np.array(H_lin).T.astype(float)
    C_lin = sp.Matrix([[C[i, j].subs([(var_sym, var_num) for var_sym, var_num in zip(sym_arr, num_arr)])
                        for i in range(7)] for j in range(7)])
    C_lin = np.array(C_lin).T.astype(float)
    Xi_lin = sp.Matrix([[Xi[i, j].subs([(var_sym, var_num) for var_sym, var_num in zip(sym_arr, num_arr)])
                        for i in range(7)] for j in range(4)])
    Xi_lin = np.array(Xi_lin).T.astype(float)


    dXidq_lin = sp.Matrix([[dXidq[i, j].subs([(var_sym, var_num) for var_sym, var_num in zip(sym_arr, num_arr)])
                            for i in range(7)] for j in range(7)])
    dXidq_lin = np.array(dXidq_lin).astype(float)

    # dCdq_lin = sp.Matrix([[dCdq[i, j].subs([(var_sym, var_num) for var_sym, var_num in zip(sym_arr, num_arr)])
    #                         for i in range(7)] for j in range(7)])
    # dCdq_lin = np.array(dCdq_lin).astype(float)
    # print(dCdq_lin)


    G_lin = sp.Matrix([[G_mat[i, j].subs([(var_sym, var_num) for var_sym, var_num in zip(sym_arr, num_arr)])
                        for i in range(7)] for j in range(7)])
    G_lin = np.array(G_lin).T.astype(float)

    iH = np.linalg.inv(H_lin)

    A = np.block([[np.zeros((7, 7)), np.eye(7)], [iH @ (dXidq_lin - G_lin), -iH @ C_lin]]).astype(float)
    B = np.vstack((np.zeros((7, 4)), iH @ Xi_lin)).astype(float)

    ctrb = control.ctrb(A, B)
    print(np.linalg.matrix_rank(ctrb))

    Q = np.diag(np.hstack((10 * np.ones(2), 100, 1 * np.ones(2), 10, 0.1 * np.ones(3), 0.2 * np.ones(3),
                           0.01, 0.05)))
    R = 50 * np.eye(4)

    S = scipy.linalg.solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ S
    return K


def linearize_quad_sys():
    mb = 0.605
    J = sp.diag(1.5e-3, 1.45e-3, 2.66e-3)
    g = 9.81
    t = sp.symbols('t')
    phi, theta, psi, rx, ry, rz, alpha, F, taux, tauy, tauz = sp.Function('phi')(t), sp.Function('theta')(t), sp.Function('psi')(t), \
                                                 sp.Function('rx')(t), sp.Function('ry')(t), sp.Function('rz')(t), \
                                                 sp.Function('alpha')(t), sp.Function('F')(t), sp.Function('taux')(t), \
                                                              sp.Function('tauy')(t), sp.Function('tauz')(t)

    q = sp.Matrix([rx, ry, rz, phi, theta, psi])
    dq = sp.diff(q, t)

    W = sp.Matrix([[sp.cos(theta)*sp.cos(psi), -sp.sin(psi), 0], [sp.cos(theta)*sp.sin(psi), sp.cos(psi), 0],
                   [-sp.sin(theta), 0, 1]])

    R = sp.Matrix([[sp.cos(psi), -sp.sin(psi), 0], [sp.sin(psi), sp.cos(psi), 0], [0, 0, 1]]) * \
        sp.Matrix([[sp.cos(theta), 0, sp.sin(theta)], [0, 1, 0], [-sp.sin(theta), 0, sp.cos(theta)]]) * \
        sp.Matrix([[1, 0, 0], [0, sp.cos(phi), -sp.sin(phi)], [0, sp.sin(phi), sp.cos(phi)]])

    Q = R.T * W

    H = sp.zeros(6)
    H[0:3, 0:3] = mb * sp.eye(3)
    H[3:6, 3:6] = Q.T * J * Q

    U = mb * g * rz

    C = sp.zeros(6)
    for i, j in zip(range(6), range(6)):
        C[i, j] = sum([0.5 * (H[i, j].diff(q[k]) + H[i, k].diff(q[j])
                              + H[j, k].diff(q[i])) * q[k].diff(t) for k in range(6)])

    N = sp.Matrix([[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    Rb = sp.eye(6)
    Rb[0:3, 0:3] = R
    Rb[3:6, 3:6] = Q.T
    Xi = Rb * N  # u = Xi * u_quad

    u = sp.Matrix([F, taux, tauy, tauz, sp.Derivative(0, t)])

    xi = sp.Matrix([q, dq])
    xi0 = sp.zeros(12, 1)
    u0 = sp.Matrix([mb * g, 0, 0, 0, 0])
    sym_arr = sp.Matrix.vstack(xi, u)
    num_arr = sp.Matrix.vstack(xi0, u0)

    H_lin = sp.Matrix([[H[i, j].subs([(var_sym, var_num) for var_sym, var_num in zip(sym_arr, num_arr)])
                        for i in range(6)] for j in range(6)])
    H_lin = np.array(H_lin).T.astype(float)
    C_lin = sp.Matrix([[C[i, j].subs([(var_sym, var_num) for var_sym, var_num in zip(sym_arr, num_arr)])
                        for i in range(6)] for j in range(6)])
    C_lin = np.array(C_lin).T.astype(float)
    Xi_lin = sp.Matrix([[Xi[i, j].subs([(var_sym, var_num) for var_sym, var_num in zip(sym_arr, num_arr)])
                        for i in range(6)] for j in range(4)])
    Xi_lin = np.array(Xi_lin).T.astype(float)

    dXidq = (Xi*sp.Matrix([F, taux, tauy, tauz])).diff(q)[:, 0, :, 0]

    dXidq_lin = sp.Matrix([[dXidq[i, j].subs([(var_sym, var_num) for var_sym, var_num in zip(sym_arr, num_arr)])
                        for i in range(6)] for j in range(6)])
    dXidq_lin = np.array(dXidq_lin).astype(float)

    G = U.diff(q)
    G_mat = sp.diff(G, q)
    G_mat = G_mat[:, 0, :, 0]
    G_lin = sp.Matrix([[G_mat[i, j].subs([(var_sym, var_num) for var_sym, var_num in zip(sym_arr, num_arr)])
                        for i in range(6)] for j in range(6)])
    G_lin = np.array(G_lin).T.astype(float)

    iH = np.linalg.inv(H_lin)

    A = np.block([[np.zeros((6, 6)), np.eye(6)], [iH @ (dXidq_lin - G_lin), -iH @ C_lin]]).astype(float)
    B = np.vstack((np.zeros((6, 4)), iH @ Xi_lin)).astype(float)

    ctrb = control.ctrb(A, B)
    print(np.linalg.matrix_rank(ctrb))

    Q = np.diag(np.hstack((4*np.ones(3), 10*np.ones(3), 2*np.ones(3), 2*np.ones(3))))
    R = 0.1 * np.eye(B.shape[1])

    S = scipy.linalg.solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ S
    return K


def deriv(x, u, m, mL, g, L, J):
    x, dx, z, dz, phi, dphi, phiL, dphiL = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]
    F, tau = u[0], u[1]
    return np.array([dx, F * sp.cos(phi - phiL) * sp.sin(phiL) / (m + mL) + m * L * sp.sin(phiL) * dphiL ** 2 / (m + mL), dz,
               -g + F * sp.cos(phi - phiL) * sp.cos(phiL) / (m + mL) + m * L * sp.cos(phiL) * dphiL ** 2 / (m + mL),
               dphi, tau / J, dphiL, F * sp.sin(phi - phiL) / m / L])


def simulate(K):
    # Simulation parameters
    T = 10
    dt = 0.005
    num_steps = int(T/dt) + 1

    # Initial conditions
    x = np.zeros((num_steps, 8))      # state: x, dx, z, dz, phi, dphi, phiL, dphiL
    u = np.zeros((num_steps, 2))      # state: x, dx, z, dz, phi, dphi, phiL, dphiL
    x[0, :] = np.array([0, 0, 0, 0, 0, 0, np.pi/10, 0])  # initial condition
    dx = np.zeros_like(x)

    for k in range(num_steps-1):
        u[k, :] = -K @ x[k, :] + np.array([1.55 * 9.81, 0])
        dx[k, :] = deriv(x[k, :], u[k, :], 1.5, 0.1, 9.81, 0.4, 0.082)
        x[k+1, :] = x[k, :] + dt * dx[k, :]
    return np.arange(0, T+dt, dt), x, dx


def format_matrix(matrix, environment="pmatrix", formatter=str):
    """Format a matrix using LaTeX syntax"""

    if not isinstance(matrix, np.ndarray):
        try:
            matrix = np.array(matrix)
        except Exception:
            raise TypeError("Could not convert to Numpy array")

    if len(shape := matrix.shape) == 1:
        matrix = matrix.reshape(1, shape[0])
    elif len(shape) > 2:
        raise ValueError("Array must be 2 dimensional")

    body_lines = [" & ".join(map(formatter, row)) for row in matrix.round(4)]

    body = "\\\\\n".join(body_lines)
    return f"""\\begin{{{environment}}}
{body}
\\end{{{environment}}}"""


if __name__ == '__main__':
    K = linearize_full_sys()
    # t, x, dx = simulate(K)
    # import matplotlib.pyplot as plt
    # plt.figure()
    # for x_ in x.T:
    #     plt.plot(t[:-1], x_, label='lab')
    # plt.legend()
    # plt.show()
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            if np.abs(K[i, j]) < 1e-4:
                K[i, j] = 0
    print(format_matrix(K.T, "bmatrix"))