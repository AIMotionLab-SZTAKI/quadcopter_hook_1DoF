import numpy as np
import casadi as ca
import scipy.linalg as si
import pickle


param_num = [0.605, 0.05, 9.81, 0.4, 1.5e-3, 1.5e-3, 2.66e-3]


def hat(a):
    mat_lst = [[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]]
    mat = ca.MX(3, 3)
    for i in range(3):
        for j in range(3):
            mat[i, j] = mat_lst[i][j]
    return mat


def eval_model():
    mb, ml, g, L, Jx, Jy, Jz = ca.MX.sym('mb'), ca.MX.sym('ml'), ca.MX.sym('g'), \
                               ca.MX.sym('L'), ca.MX.sym('Jx'), ca.MX.sym('Jy'), \
                               ca.MX.sym('Jz')
    params = [mb, ml, g, L, Jx, Jy, Jz]

    J = ca.MX(3, 3)
    J[0, 0], J[1, 1], J[2, 2] = Jx, Jy, Jz
    phi, theta, psi, rx, ry, rz, alpha, F, taux, tauy, tauz = ca.MX.sym('phi'), ca.MX.sym('theta'), ca.MX.sym('psi'), \
                                                              ca.MX.sym('rx'), ca.MX.sym('ry'), ca.MX.sym('rz'), \
                                                              ca.MX.sym('alpha'), ca.MX.sym('F'), ca.MX.sym('taux'), \
                                                              ca.MX.sym('tauy'), ca.MX.sym('tauz')
    dphi, dtheta, dpsi, drx, dry, drz, dalpha = ca.MX.sym('dphi'), ca.MX.sym('dtheta'), ca.MX.sym('dpsi'), \
                                                ca.MX.sym('drx'), ca.MX.sym('dry'), ca.MX.sym('drz'), \
                                                ca.MX.sym('dalpha')
    r = ca.vertcat(rx, ry, rz)
    q_lst = [rx, ry, rz, phi, theta, psi, alpha]
    q = ca.vertcat(*q_lst)

    dq_lst = [drx, dry, drz, dphi, dtheta, dpsi, dalpha]
    dq = ca.vertcat(*dq_lst)

    u_lst = [F, taux, tauy, tauz]
    u = ca.vertcat(*u_lst)

    W_lst = [[ca.cos(theta) * ca.cos(psi), -ca.sin(psi), 0], [ca.cos(theta) * ca.sin(psi), ca.cos(psi), 0],
             [-ca.sin(theta), 0, 1]]
    W = ca.vertcat(ca.horzcat(*W_lst[0]), ca.horzcat(*W_lst[1]), ca.horzcat(*W_lst[2]))

    R_lst = [[ca.cos(psi) * ca.cos(theta), ca.cos(psi) * ca.sin(phi) * ca.sin(theta) - ca.cos(phi) * ca.sin(psi),
              ca.sin(phi) * ca.sin(psi) + ca.cos(phi) * ca.cos(psi) * ca.sin(theta)],
             [ca.cos(theta) * ca.sin(psi), ca.cos(phi) * ca.cos(psi) + ca.sin(phi) * ca.sin(theta) * ca.sin(psi),
              ca.cos(phi) * ca.sin(theta) * ca.sin(psi) - ca.cos(psi) * ca.sin(phi)],
             [-ca.sin(theta), ca.sin(phi) * ca.cos(theta), ca.cos(phi) * ca.cos(theta)]]
    R = ca.vertcat(ca.horzcat(*R_lst[0]), ca.horzcat(*R_lst[1]), ca.horzcat(*R_lst[2]))

    # Q = R.T * W
    Q_lst = [[1, 0, -ca.sin(theta)], [0, ca.cos(phi), ca.sin(phi) * ca.cos(theta)],
             [0, -ca.sin(phi), ca.cos(phi) * ca.cos(theta)]]
    Q = ca.vertcat(ca.horzcat(*Q_lst[0]), ca.horzcat(*Q_lst[1]), ca.horzcat(*Q_lst[2]))

    invQ_lst = [[1, ca.sin(phi) * ca.tan(theta), ca.cos(phi) * ca.tan(theta)], [0, ca.cos(phi), -ca.sin(phi)],
                [0, ca.sin(phi) / ca.cos(theta), ca.cos(phi) / ca.cos(theta)]]
    invQ = ca.vertcat(ca.horzcat(*invQ_lst[0]), ca.horzcat(*invQ_lst[1]), ca.horzcat(*invQ_lst[2]))

    dQ_lst = [[0, 0, -ca.cos(theta)*dtheta],
              [0, -ca.sin(phi)*dphi, ca.cos(phi) * ca.cos(theta) * dphi - ca.sin(phi) * ca.sin(theta) * dtheta],
              [0, -ca.cos(phi)*dphi, -ca.sin(phi) * ca.cos(theta) * dphi - ca.cos(phi) * ca.sin(theta) * dtheta]]
    dQ = ca.vertcat(ca.horzcat(*dQ_lst[0]), ca.horzcat(*dQ_lst[1]), ca.horzcat(*dQ_lst[2]))

    peb = ca.vertcat(-L * ca.sin(alpha), 0, -L * ca.cos(alpha))

    Jp = ca.vertcat(-L * ca.cos(alpha), 0, L * ca.sin(alpha))

    H = ca.MX(7, 7)
    H[0:3, 0:3] = (mb + ml) * ca.MX_eye(3)
    H[3:6, 3:6] = ml * W.T @ hat(R @ peb).T @ hat(R @ peb) @ W + Q.T @ J @ Q
    H[6, 6] = ml * L ** 2
    H[0:3, 3:6] = -ml * hat(R @ peb) @ W
    H[3:6, 0:3] = H[0:3, 3:6].T
    H[0:3, 6] = ml * R @ Jp
    H[6, 0:3] = H[0:3, 6].T
    H[3:6, 6] = -ml * W.T @ hat(R @ peb).T @ R @ Jp
    H[6, 3:6] = H[3:6, 6].T

    U = mb * g * rz + ml * g * (r + R @ peb)[2]
    G = ca.gradient(U, q)

    C = ca.MX(7, 7)
    for i, j in zip(range(7), range(7)):
        C[i, j] = sum([0.5 * (ca.gradient(H[i, j], q[k]) + ca.gradient(H[i, k], q[j])
                              + ca.gradient(H[j, k], q[i])) * dq[k] for k in range(7)])

    N = ca.MX(7, 4)
    N[2, 0] = 1
    N[3, 1] = 1
    N[4, 2] = 1
    N[5, 3] = 1

    Rb = ca.MX_eye(7)
    Rb[0:3, 0:3] = R
    Rb[3:6, 3:6] = Q.T
    Xi = Rb @ N

    rhs = ca.inv(H) @ (Xi @ u - C @ dq - G)
    f1_rhs = ca.Function('f1', q_lst + dq_lst + u_lst + params, [ca.jacobian(rhs, q)])
    f2_rhs = ca.Function('f2', q_lst + dq_lst + u_lst + params, [ca.jacobian(rhs, dq)])
    f3_rhs = ca.Function('f3', q_lst + dq_lst + u_lst + params, [ca.jacobian(rhs, u)])
    f_rhs = ca.Function('f', q_lst + dq_lst + u_lst + params, [rhs])
    fH = ca.Function('fH', q_lst + dq_lst + u_lst + params, [H])
    fC = ca.Function('fC', q_lst + dq_lst + u_lst + params, [C])
    fXi = ca.Function('fXi', q_lst + dq_lst + u_lst + params, [Xi])
    fG = ca.Function('fG', q_lst + dq_lst + u_lst + params, [G])
    fQ = ca.Function('fH', q_lst + dq_lst + u_lst + params, [Q])
    fdQ = ca.Function('fdQ', q_lst + dq_lst + u_lst + params, [dQ])
    fR = ca.Function('fR', q_lst + dq_lst + u_lst + params, [R])
    finvQ = ca.Function('finvQ', q_lst + dq_lst + u_lst + params, [invQ])

    functions_dict = dict(f1=f1_rhs, f2=f2_rhs, f3=f3_rhs, f=f_rhs, fH=fH, fC=fC, fXi=fXi, fG=fG, fQ=fQ,
                          fdQ=fdQ, fR=fR, finvQ=finvQ)

    return functions_dict


def linearize_model(f_rhs: ca.Function, f1_rhs: ca.Function, f2_rhs: ca.Function, f3_rhs: ca.Function):
    setpoint = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (param_num[0] + param_num[1])*param_num[2], 0, 0, 0]
    A = np.hstack([f1_rhs(*(setpoint + param_num)), f2_rhs(*(setpoint + param_num))])
    A = np.vstack([np.hstack([np.zeros((7, 7)), np.eye(7)]), A])
    B = np.array(f3_rhs(*(setpoint + param_num)))
    B = np.vstack([np.zeros((7, 4)), B])
    return A, B


def compute_lqr(A, B):
    Q = np.diag(np.hstack((10 * np.ones(2), 100, 1 * np.ones(2), 10, 0.1 * np.ones(3), 1 * np.ones(3),
                           0.01, 0.05)))
    R = np.diag((5, 20, 20, 20))

    S = si.solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ S
    return K


def test_eval_time(f_rhs, f1_rhs, f2_rhs):
    X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15 = np.meshgrid(*(15 * [np.linspace(-0.2, 0.2, 2)]))

    # Iterate over all grid points
    it = np.nditer([X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15], flags=['multi_index'])

    while not it.finished:
        idx = it.multi_index

        setpoint = [0, 0, 0, X1[idx], X2[idx], X3[idx], X4[idx], X5[idx], X6[idx], X7[idx], X8[idx], X9[idx],
                    X10[idx], X11[idx], X12[idx], X13[idx], X14[idx], X15[idx]]

        f_cur = f_rhs(*(setpoint + param_num))
        f_cur1 = f1_rhs(*(setpoint + param_num))
        f_cur2 = f2_rhs(*(setpoint + param_num))

        # Move to the next grid point
        it.iternext()


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

    body_lines = [" & ".join(map(formatter, row)) for row in matrix.round(2)]

    body = "\\\\\n".join(body_lines)
    return f"""\\begin{{{environment}}}
{body}
\\end{{{environment}}}"""


if __name__ == "__main__":
    model = eval_model()
    A, B = linearize_model(model['f'], model['f1'], model['f2'], model['f3'])
    K = compute_lqr(A, B)
    A_cl = A - B @ K
