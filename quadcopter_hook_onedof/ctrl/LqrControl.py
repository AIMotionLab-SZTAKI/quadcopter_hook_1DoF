# LQR control to stabilize the drone-payload system.

import numpy as np
from scipy.spatial.transform import Rotation
import quadcopter_hook_onedof.ctrl.casadi_model as cm


class LqrControl:
    def __init__(self, model):
        self.model = model
        self.drone_mass = 0.605
        self.payload_mass = model.body_mass[4]
        self.mass = self.drone_mass + self.payload_mass
        self.g = 9.81
        self.inertia = np.diag([1.5e-3, 1.5e-3, 2.66e-3])
        self.L = 0.4
        self.K = self.compute_lqr()
        for i in range(self.K.shape[0]):
            for j in range(self.K.shape[1]):
                if np.abs(self.K[i, j]) < 1e-8:
                    self.K[i, j] = 0
        ordering = [0, 1, 2, 7, 8, 9, 3, 4, 5, 10, 11, 12, 6, 13]
        # idx = np.empty_like(ordering)
        # idx[ordering] = np.arange(len(ordering))
        # K_rearranged = self.K[:, ordering]  # return a rearranged copy

        # with open("lqr_params_c.txt", "w") as f:
        #     for i in range(K_rearranged.shape[0]):
        #         for j in range(K_rearranged.shape[1]):
        #             if np.abs(K_rearranged[i, j]) > 1e-8:
        #                 f.write(f'static float K{i+1}{j+1} = {K_rearranged[i, j].round(4)};\n')
        # print(format_matrix(K_rearranged.T, "bmatrix"))

    def compute_control(self,
                        cur_pos,
                        cur_quat,
                        cur_vel,
                        cur_ang_vel,
                        cur_alpha,
                        cur_dalpha,
                        target_pos,
                        target_rpy=np.zeros(3),
                        target_vel=np.zeros(3),
                        target_rpy_rates=None):
        cur_quat = np.roll(cur_quat, -1)
        cur_eul = Rotation.from_quat(cur_quat).as_euler('ZYX')[-1::-1]
        while cur_eul[2] - target_rpy[2] > np.pi:
            cur_eul[2] -= 2 * np.pi
        while cur_eul[2] - target_rpy[2] < -np.pi:
            cur_eul[2] += 2 * np.pi
        pos_err = cur_pos - target_pos
        yaw = target_rpy[2]
        yaw_tran = np.array([[np.cos(yaw), np.sin(yaw), 0],[-np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        pos_err = yaw_tran @ pos_err
        vel_err = cur_vel - target_vel
        vel_err = yaw_tran @ vel_err
        error = np.hstack((pos_err,
                           cur_eul - target_rpy,
                           cur_alpha,
                           vel_err,
                           cur_ang_vel,
                           cur_dalpha))
        ctrl = -self.K @ error
        ctrl[0] = self.mass * self.g + ctrl[0]
        return ctrl

    def compute_lqr(self):
        model = cm.eval_model()
        A, B = cm.linearize_model(model['f'], model['f1'], model['f2'], model['f3'])
        K = cm.compute_lqr(A, B)
        return K


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
    ctrl = LqrControl(None)
