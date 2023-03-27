# Planar LQR control to stabilize the payload.

import numpy as np
from ctrl.compute_lqr import linearize_sys


class PlanarLQRControl:
    def __init__(self, model):
        self.model = model
        self.K = linearize_sys()
        print(self.K)
        self.mass = 0.605
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