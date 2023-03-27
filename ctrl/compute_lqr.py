import numpy as np
import sympy as sp
import scipy as si
import control


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
    f = sp.Matrix([dx, F*sp.cos(phi-phiL)*sp.sin(phiL)/(m+mL) + m*L*sp.sin(phiL)*dphiL**2/(m+mL), dz,
                   -g + F * sp.cos(phi - phiL) * sp.cos(phiL) / (m + mL) + m * L * sp.cos(phiL) * dphiL ** 2 / (m + mL),
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
    param_num = [0.605, 0.1, 9.81, 0.4, 1.5e-3, 0]
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
    Q = np.diag([1, 1, 1, 1, 1, 1, 1, 1])
    R = 1e-2*np.eye(2)
    K, S, E = control.dlqr(A_d, B_d, Q, R, method='scipy')
    return K


def linearize_full_sys():
    mb = 1.5
    ml = 0.05
    J = sp.diag(0.082, 0.085, 0.138)
    g = 9.81
    L = 0.4
    t = sp.symbols('t')
    phi, theta, psi, rx, ry, rz, alpha = sp.Function('phi')(t), sp.Function('theta')(t), sp.Function('psi')(t), \
                                         sp.Function('rx')(t), sp.Function('ry')(t), sp.Function('rz')(t), \
                                         sp.Function('alpha')(t)
    dphi, dtheta, dpsi, drx, dry, drz, dalpha = sp.symbols("dphi, dtheta, dpsi, drx, dry, drz, dalpha")
    lam = sp.Matrix([phi, theta, psi])
    r = sp.Matrix([rx, ry, rz])
    q = sp.Matrix([rx, ry, rz, phi, theta, psi, alpha])
    S_phi = sp.sin(phi)
    S_theta = sp.sin(theta)
    S_psi = sp.sin(psi)
    C_phi = sp.cos(phi)
    C_theta = sp.cos(theta)
    C_psi = sp.cos(psi)

    W = sp.Matrix([[1, 0, -S_theta], [0, C_phi, C_theta*S_phi], [0, -S_phi, C_theta*C_phi]])

    R = (sp.Matrix([[1, 0, 0], [0, C_phi, S_phi], [0, -S_phi, C_phi]]) *
         sp.Matrix([[C_theta, 0, -S_theta], [0, 1, 0], [S_theta, 0, C_theta]]) *
         sp.Matrix([[C_psi, S_psi, 0], [-S_psi, C_psi, 0], [0, 0, 1]])).T

    Q = R.T * W

    Jp = sp.Matrix([L*sp.cos(alpha), 0, L*sp.sin(alpha)])
    Reb = sp.Matrix([[sp.cos(alpha), 0, -sp.sin(alpha)], [0, 1, 0], [sp.sin(alpha), 0, sp.cos(alpha)]])  # TODO transpose?
    peb = sp.Matrix([L*sp.sin(alpha), 0, -L*sp.cos(alpha)])

    H = sp.zeros(7)
    H[0:3, 0:3] = (mb + ml) * sp.eye(3)
    H[3:6, 3:6] = Q.T * J * Q + ml * W.T * hat(R * peb).T * hat(R * peb).T * W
    H[6, 6] = ml * L ** 2
    H[0:3, 3:6] = -ml * hat(R * peb) * W
    H[3:6, 0:3] = H[0:3, 3:6].T
    H[0:3, 6] = ml * R * Jp
    H[6, 0:3] = H[0:3, 6].T
    H[3:6, 6] = -ml * W.T * hat(R * peb).T * R * Jp
    H[6, 3:6] = H[3:6, 6].T

    U = mb * g * rz + ml * g * (r + R * peb)[2]

    G = U.diff(q)

    C = sp.zeros(7)
    for i, j in zip(range(7), range(7)):
        C[i, j] = sum([0.5 * (H[i, j].diff(q[k]) + H[i, k].diff(q[j])
                              + H[j, k].diff(q[i])) * q[k].diff(t) for k in range(7)])

    N = sp.Matrix([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 1]])
    Rb = sp.eye(7)
    Rb[0:3, 0:3] = R
    Rb[3:6, 3:6] = Q.T
    Xi = Rb * N  # u = Xi * u_quad

    q_arr = [rx, rz, theta, alpha]
    dq_arr = [q_.diff(t) for q_ in q_arr]
    state_arr = q_arr + dq_arr

    print('ye')


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


if __name__ == '__main__':
    K = linearize_sys()
    t, x, dx = simulate(K)
    import matplotlib.pyplot as plt
    plt.figure()
    for x_ in x.T:
        plt.plot(t[:-1], x_, label='lab')
    plt.legend()
    plt.show()
