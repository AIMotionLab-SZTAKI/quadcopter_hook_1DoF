import numpy as np
from planning.traj_opt_min_time import construct


if __name__ == '__main__':
    init_pos_rel = [-0.76, 2.13, 1.0 - 0.615, 0]
    load_target_rel = [0.76, 2.13, 0]
    construct(init_pos=init_pos_rel, load_target=load_target_rel, load_mass=0.1, plot_result=True)
