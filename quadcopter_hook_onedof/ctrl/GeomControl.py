import numpy as np
from scipy.spatial.transform import Rotation


class GeomControl:
    """
    Geometric ctrl class for quadcopters.
    """

    ################################################################################

    def __init__(self, model, data, drone_type='cf2'):
        if drone_type == 'cf2':
            self.k_r = 1
            self.k_v = 0.5
            self.k_R = 0.1
            self.k_w = 0.002
            self.inertia = np.diag([1.4e-5, 1.4e-5, 2.1e-5])
            self.mass = 0.028
        elif drone_type == 'large_quad':
            self.k_r = 6
            self.k_v = 3
            self.k_R = 1.7
            self.k_w = 0.2
            self.inertia = np.diag([1.5e-3, 1.45e-3, 2.66e-3])
            self.mass = 0.605

        # elif drone_type == 'large_quad':
        #     self.k_r = 25
        #     self.k_v = 16
        #     self.k_R = 15
        #     self.k_w = 2.54
        #     self.inertia = np.diag([0.082, 0.085, 0.138])
        #     self.mass = 4.34

        else:
            raise NotImplementedError

        self.model = model
        self.data = data
        self.gravity = 9.81

        # optional integrator term
        self.k_i = 0
        self.dt = 0.02
        self.int_pos_e = np.zeros(3)

        self.reset()

    ################################################################################

    def reset(self):
        self.int_pos_e = np.zeros(3)
        pass

    ################################################################################

    def compute_pos_control(self,
                            cur_pos,
                            cur_quat,
                            cur_vel,
                            cur_ang_vel,
                            target_pos,
                            target_rpy=np.zeros(3),
                            target_vel=np.zeros(3),
                            target_rpy_rates=np.zeros(3)
                            ):
        """
        Geometric ctrl of quadcopters: position ctrl based on reference position and velocity.
        """
        pos_e = cur_pos - target_pos
        vel_e = cur_vel - target_vel
        target_acc = np.zeros(3)
        target_yaw = target_rpy[2]
        self.int_pos_e = self.int_pos_e + self.dt * pos_e

        A = -self.k_r*pos_e - self.k_v*vel_e - self.k_i*self.int_pos_e + \
            self.mass*self.gravity*np.array([0, 0, 1]) + self.mass*target_acc + self._mu_r(pos_e, vel_e)
        r3 = A / np.linalg.norm(A)
        if np.abs(target_yaw) < 1e-3:  # speed up cross product if yaw target is zero
            cross_temp = self._my_cross(r3)
            r2 = cross_temp / np.linalg.norm(cross_temp)
            r1 = self._my_cross_2(r2, r3)
        else:
            cross_temp = np.cross(r3, np.array([np.cos(target_yaw), np.sin(target_yaw), 0]))
            r2 = cross_temp / np.linalg.norm(cross_temp)
            r1 = np.cross(r2, r3)
        target_rotation = np.array([r1, r2, r3]).transpose()

        # Convert quaternion from scalar first to scalar last format
        cur_quat = np.roll(cur_quat, -1)
        cur_rotation = Rotation.from_quat(cur_quat).as_matrix()
        rot_e = 1 / 2 * self._veemap(np.dot((target_rotation.transpose()), cur_rotation)
                                     - np.dot(cur_rotation.transpose(), target_rotation))
        if np.isnan(np.sum(rot_e)):
            rot_e = np.zeros(3)
        ang_vel_e = cur_ang_vel - np.dot(np.dot(cur_rotation.transpose(), target_rotation), target_rpy_rates)
        target_torques = -self.k_R * rot_e - self.k_w * ang_vel_e + np.cross(cur_ang_vel,
                         np.dot(self.inertia, cur_ang_vel)) - self.inertia @ self._mu_R(cur_quat, cur_ang_vel, rot_e, ang_vel_e)
        thrust = np.dot(A, np.dot(cur_rotation, np.array([0, 0, 1])))

        ctrl = np.zeros(4)
        ctrl[0] = np.clip(thrust, 0, self.model.actuator_ctrlrange[0, 1])
        ctrl[1:4] = target_torques

        return ctrl

    def compute_att_control(self,
                            cur_pos,
                            cur_quat,
                            cur_vel,
                            cur_ang_vel,
                            target_pos,
                            target_vel,
                            target_acc,
                            target_quat,
                            target_quat_vel,
                            ):
        """
        Geometric ctrl of quadcopters: attitude ctrl based on reference quaternion, quaternion derivative,
        position and velocity. Quaternion should be in scalar first format.
        """
        # Convert quaternion from scalar first to scalar last format
        cur_quat = np.roll(cur_quat, -1)
        target_quat = np.roll(target_quat, -1)
        # omega = Im{2*conj(q)*dq}
        target_ang_vel = (2 * self._quat_mult(self._quat_conj(target_quat), target_quat_vel))[0:3]
        target_rotation = Rotation.from_quat(target_quat).as_matrix()
        cur_rotation = Rotation.from_quat(cur_quat).as_matrix()
        rot_e = 1 / 2 * self._veemap(np.dot((target_rotation.transpose()), cur_rotation)
                                     - np.dot(cur_rotation.transpose(), target_rotation))
        ang_vel_e = cur_ang_vel - np.dot(np.dot(cur_rotation.transpose(), target_rotation), target_ang_vel)
        target_torques = -self.k_R * rot_e - self.k_w * ang_vel_e + np.cross(cur_ang_vel,
                                                                             np.dot(self.inertia, cur_ang_vel))
        pos_e = cur_pos - target_pos
        vel_e = cur_vel - target_vel
        e3 = np.array([0, 0, 1])
        den = np.dot(e3, np.dot(cur_rotation, e3))
        if np.abs(den) > 1e-7:
            thrust = (-self.k_r * pos_e[2] - self.k_v * vel_e[2] + self.mass * self.gravity +
                      self.mass * target_acc[2]) / den
        else:
            print('den is close to zero')
            thrust = 0

        # psi = 0.5 * np.trace(np.eye(3) - target_rotation.T @ cur_rotation)

        ctrl = np.zeros(4)
        ctrl[0] = np.clip(thrust, 0, 0.64)
        ctrl[1:4] = target_torques

        return ctrl

    def _mu_r(self, pos_e, vel_e):
        return np.zeros(3)

    def _mu_R(self, cur_quat, cur_ang_vel, rot_e, ang_vel_e):
        return np.zeros(3)

    @staticmethod
    def _veemap(mat):
        a = np.zeros(3)
        a[0] = mat[2, 1]
        a[1] = mat[0, 2]
        a[2] = mat[1, 0]
        return a

    @staticmethod
    def _hatmap(a):
        mat = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        return mat

    @staticmethod
    def _quat_mult(quaternion1, quaternion0):
        """Multiply two quaternions in scalar last form"""
        x0, y0, z0, w0 = quaternion0
        x1, y1, z1, w1 = quaternion1
        return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                         x1 * w0 + y0 * z0 - z1 * y0 + w1 * x0,
                         -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                         x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

    @staticmethod
    def _quat_conj(quat):
        """Return conjugate of a quaternion in scalar last form"""
        return np.array([-quat[0], -quat[1], -quat[2], quat[3]])

    def stability_analysis(self, k_r, k_v, k_R, k_w, c1, c2, J, m, eps=None):

        ######## Stability analysis of geometric ctrl based on the original paper #################

        Lmax = max(np.linalg.eigvals(J))
        Lmin = min(np.linalg.eigvals(J))
        psi1 = 0.01
        alpha = np.sqrt(psi1 * (2 - psi1))
        e_rmax = 0.001  # max{||e_v(0)||, B/(k_v*(1-alpha))}, TODO: see (23), (24)
        B = 0.0001  # B > ||-mge_3 + m\ddot{x}||  TODO: calculate B, see (16)
        W1 = np.array([[c1 * k_r / m * (1 - alpha), -c1 * k_v / (2 * m) * (1 + alpha)],
                       [-c1 * k_v / (2 * m) * (1 + alpha), k_v * (1 - alpha) - c1]])
        W12 = np.array([[c1 / m * B, 0], [B + k_r * e_rmax, 0]])
        W2 = np.array([[c2 * k_R / Lmax, -c2 * k_w / (2 * Lmin)], [-c2 * k_w / (2 * Lmin), k_w - c2]])
        exp1 = min([k_v * (1 - alpha),
                    4 * m * k_r * k_v * (1 - alpha) ** 2 / (k_v ** 2 * (1 + alpha) ** 2 + 4 * m * k_r * (1 - alpha)),
                    np.sqrt(k_r * m)])
        crit1 = c1 < exp1
        exp2 = min([k_w, 4 * k_w * k_R * Lmin ** 2 / (k_w ** 2 * Lmax + 4 * k_R * Lmin ** 2), np.sqrt(k_R * Lmin)])
        crit2 = c2 < exp2
        exp3 = [min(np.linalg.eigvals(W2)), 4 * np.linalg.norm(W12, ord=2) ** 2 / min(np.linalg.eigvals(W1))]
        crit3 = exp3[0] > exp3[1]

        return crit1, crit2, crit3

    @staticmethod
    def _my_cross(r3):
        '''
        Simplify cross product if the second vector is [0, 0, 1].
        '''
        return np.array([0, r3[2], -r3[1]])

    @staticmethod
    def _my_cross_2(a, b):
        '''
        Simplify cross product if the first vector is [0, a2, a3].
        '''
        return np.array([a[1]*b[2]-a[2]*b[1], a[2]*b[0], -a[1]*b[0]])
