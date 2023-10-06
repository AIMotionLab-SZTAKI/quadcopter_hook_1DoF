# Planar LQR control to stabilize the payload.

import numpy as np
from quadcopter_hook_onedof.ctrl.compute_lqr import linearize_sys


class PlanarLqrControl:
    def __init__(self, model):
        self.model = model
        self.K = linearize_sys()
        for i in range(self.K.shape[0]):
            for j in range(self.K.shape[1]):
                if np.abs(self.K[i, j]) < 1e-8:
                    self.K[i, j] = 0
        print(format_matrix(self.K.T, "bmatrix"))
        self.mass = 0.605 + 0.02
        self.g = 9.81

    def compute_control(self, x_L, v_L, phi_L, dphi_L, phi_Q, dphi_Q, target_x_L=0):
        e_x = x_L - target_x_L
        e_v = v_L
        e_L = phi_L
        e_vL = dphi_L
        e_Q = phi_Q
        e_vQ = dphi_Q
        thrust_and_torque = -self.K @ np.array([e_x[0], e_v[0], e_x[1], e_v[1], e_Q, e_vQ, e_L, e_vL])

        ctrl = np.zeros(4)
        ctrl[0] = np.clip(self.mass*self.g + thrust_and_torque[0], 0, self.model.actuator_ctrlrange[0, 1])
        ctrl[2] = thrust_and_torque[1]
        return ctrl


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