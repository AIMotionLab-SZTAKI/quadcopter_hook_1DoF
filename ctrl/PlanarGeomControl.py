# Planar geometric control based on Sreenath, et. al. 2013: Trajectory Generation and Control of a Quadrotor with a
# Cable-Suspended Load – A Differentially-Flat Hybrid System
# Implemented, but does not work yet.

import numpy as np


class PlanarGeomControl:
    def __init__(self, model):
        self.model = model
        self.k_px = 20
        self.k_dx = 1
        self.k_pQ = 20
        self.k_dQ = 0.1
        self.k_pL = 0.01
        self.k_dL = 0.00001
        self.inertia = np.diag([0.082, 0.085, 0.138])
        self.m_Q = 1.5
        self.m_L = 0.05
        self.g = 9.81
        self.e3 = np.array([0, 1])

    def compute_control(self, x_L, v_L, phi_L, dphi_L, phi_Q, dphi_Q, target_x_L=0):
        e_x = x_L - target_x_L
        e_v = v_L
        A = -self.k_px * e_x - self.k_dx * e_v + self.m_L * self.g * self.e3
        B = self.m_Q * self.g * self.e3
        R = np.array([[np.cos(phi_Q), -np.sin(phi_Q)],[np.sin(phi_Q), np.cos(phi_Q)]])
        thrust = (A + B) @ R @ self.e3
        target_phi_L = np.arctan2(A[0], A[1])
        e_L = phi_L - target_phi_L
        e_vL = dphi_L
        # print( - self.k_pL * e_L - self.k_dL * e_vL)
        target_phi_Q = np.arcsin( - self.k_pL * e_L - self.k_dL * e_vL)# + phi_L
        e_Q = phi_Q - target_phi_Q
        e_vQ = dphi_Q
        torque = self.inertia[1, 1] * (-self.k_pQ * e_Q - self.k_dQ * e_vQ)

        ctrl = np.zeros(4)
        ctrl[0] = np.clip(thrust, 0, self.model.actuator_ctrlrange[0, 1])
        ctrl[2] = torque
        print(torque)
        return ctrl
