import numpy as np
# from quadcopter_hook_onedof.planning.demo_traj import construct
from quadcopter_hook_onedof.planning.traj_opt_min_time import construct


if __name__ == '__main__':

    # Input: payload initial position, yaw + drone initial position, yaw + payload final position
    init_pos_drone = np.array([-0.76, 0.7, 1.1])  # Drone position after takeoff
    init_yaw_drone = -np.pi/2  # Drone yaw at takeoff
    init_pos_load = np.array([0, 0, 0.67])  # Load initial position, only z coordinate is offset
    init_yaw_load = np.pi  # Load initial orientation
    target_pos_load = np.array([0.76, 0.7, 0.67])  # Load target position
    target_yaw_load = 0  # Load target yaw

    while init_yaw_load - init_yaw_drone > np.pi:
        init_yaw_load -= 2*np.pi
    while init_yaw_load - init_yaw_drone < -np.pi:
        init_yaw_load += 2 * np.pi
    while target_yaw_load - init_yaw_load > np.pi:
        target_yaw_load -= 2*np.pi
    while target_yaw_load - init_yaw_load < -np.pi:
        target_yaw_load += 2*np.pi

    # Convert to relative vectors
    R_yaw = np.array([[np.cos(init_yaw_load), -np.sin(init_yaw_load), 0],
                      [np.sin(init_yaw_load), np.cos(init_yaw_load), 0], [0, 0, 1]])
    init_pos_drone_rel = (np.linalg.inv(R_yaw) @ (init_pos_drone - init_pos_load)).tolist() + [init_yaw_drone]
    target_pos_load_rel = np.linalg.inv(R_yaw) @ (target_pos_load - init_pos_load)

    construct(init_pos=init_pos_drone_rel, load_target=target_pos_load_rel, load_mass=0.1, plot_result=True,
              save_splines=False, save_path=None, init_pos_abs=init_pos_load, load_init_yaw=init_yaw_load,
              load_target_yaw=target_yaw_load)
