import pickle

import mujoco
import glfw
import os
import numpy as np
from ctrl.GeomControl import GeomControl
from ctrl.RobustGeomControl import RobustGeomControl
from ctrl.PlanarLQRControl import PlanarLQRControl
import time
from assets.util import sync
from scipy.spatial.transform import Rotation
from assets.logger import Logger
from matplotlib import pyplot as plt
from planning.traj_opt_min_time import construct, plot_3d_trajectory


if __name__ == "__main__":
    # Trajectory parameters
    init_pos_rel = [-1.5, 0, 0.5, 0]
    load_target_rel = [1.0, 1.5, 0]
    construct(init_pos_rel, load_target_rel, plot_result=True)

    init_pos_rel = [-1, 1, 0.8, 0]
    load_target_rel = [1.2, 0, 0]
    construct(init_pos_rel, load_target_rel, plot_result=True)

    init_pos_rel = [0, -1, 0.6, 0]
    load_target_rel = [1.5, 0.5, 0]
    construct(init_pos_rel, load_target_rel, plot_result=True)

    init_pos_rel = [-0.5, 1.5, 1, 0]
    load_target_rel = [1.5, 1.5, 0]
    construct(init_pos_rel, load_target_rel, plot_result=True)
