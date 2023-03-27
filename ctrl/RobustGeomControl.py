import numpy as np
from ctrl.GeomControl import GeomControl


class RobustGeomControl(GeomControl):
    def __init__(self, model, data, drone_type='cf2'):
        super().__init__(model, data, drone_type=drone_type)
        self.delta_r = 1  # 0.01715
        self.delta_R = 0
        self.tau = 3
        self.c1 = 1
        self.c2 = 0.001
        self.eps_r = 1e-4
        self.eps_R = 1e-4
        self.Jinv = np.linalg.inv(self.inertia)

    def stability_analysis(self, k_r, k_v, k_R, k_w, c1, c2, J, m, eps=None):
        eps_r, eps_R = eps[0], eps[1]
        Lmax = max(J)
        Lmin = min(J)
        psi1 = 0.1
        alpha = np.sqrt(psi1*(2-psi1))
        e_vmax = 0  # max{||e_v(0)||, B/(k_v*(1-alpha))}, TODO: see (23), (24)
        e_rmax = 0.001  # TODO
        B = 0.001  # B > ||-mge_3 + m\ddot{x}||  TODO: calculate B, see (16)
        W1 = np.array([[c1*k_r/m*(1-alpha), -c1*k_v/(2*m)*(1+alpha)], [-c1*k_v/(2*m)*(1+alpha), k_v*(1-alpha)-c1]])
        W12 = np.array([[c1/m*(B+self.delta_r), 0], [B+self.delta_r+k_r*e_rmax, 0]])
        W2 = np.array([[c2*k_R/Lmax, -c2*k_w/(2*Lmin)], [-c2*k_w/(2*Lmin), k_w-c2]])
        W = np.array([[min(np.linalg.eigvals(W1)), -0.5*np.linalg.norm(W12, ord=2)],
                      [-0.5*np.linalg.norm(W12, ord=2), min(np.linalg.eigvals(W2))]])
        M11 = 0.5*np.array([[k_r, -c1], [-c1, m]])
        M12 = 0.5*np.array([[k_r, c1], [c1, m]])
        M21 = 0.5*np.array([[k_R, -c2], [-c2, Lmin]])
        M22 = 0.5*np.array([[2*k_R/(2-psi1), c2], [c2, Lmax]])

        exp1 = min([k_v*(1-alpha), 4*m*k_r*k_v*(1-alpha)**2/(k_v**2*(1+alpha)**2+4*m*k_r*(1-alpha)), np.sqrt(k_r*m)])
        crit1 = c1 < exp1
        exp2 = min([k_w, 4*k_w*k_R*Lmin**2/(k_w**2*Lmax+4*k_R*Lmin**2), np.sqrt(k_R*Lmin)])
        crit2 = c2 < exp2
        exp3 = [min(np.linalg.eigvals(W2)), np.linalg.norm(W12, ord=2)**2/(4*min(np.linalg.eigvals(W1)))]
        crit3 = exp3[0] > exp3[1]
        exp4 = min([min(np.linalg.eigvals(M11)), min(np.linalg.eigvals(M21))])*min([e_rmax**2, psi1*(2-psi1)]) / \
                max([max(np.linalg.eigvals(M12)), max(np.linalg.eigvals(M22))]) * min(np.linalg.eigvals(W))
        crit4 = exp4 > eps_r + eps_R

        return crit1, crit2, crit3, exp4

    def _mu_r(self, pos_e, vel_e):
        e_B = vel_e + self.c1 / self.mass * pos_e
        return -self.delta_r ** (self.tau + 2) * e_B * np.linalg.norm(e_B) ** self.tau / (self.delta_r ** (self.tau + 1)
                * np.linalg.norm(e_B) ** (self.tau + 1) + self.eps_r ** (self.tau + 1))

    def _mu_R(self, cur_quat, cur_ang_vel, rot_e, ang_vel_e):
        e_A = ang_vel_e + self.c2 * self.Jinv @ rot_e
        return -self.delta_R ** 2 * e_A / (self.delta_R * np.linalg.norm(e_A) + self.eps_R)
