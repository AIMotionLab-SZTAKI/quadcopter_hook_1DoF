import pickle

import mujoco
import glfw
import os
import numpy as np
from quadcopter_hook_onedof.ctrl.GeomControl import GeomControl
from quadcopter_hook_onedof.ctrl.RobustGeomControl import RobustGeomControl
from quadcopter_hook_onedof.ctrl.PlanarLqrControl import PlanarLqrControl
import time
from quadcopter_hook_onedof.assets.util import sync
from scipy.spatial.transform import Rotation
from quadcopter_hook_onedof.assets.logger import Logger
from matplotlib import pyplot as plt
from quadcopter_hook_onedof.planning.traj_opt_min_time import construct, plot_3d_trajectory


def main():
    # Reading model data
    print(f'Working directory:  {os.getcwd()}\n')
    model = mujoco.MjModel.from_xml_path("../assets/hook_scenario.xml")
    data = mujoco.MjData(model)

    # Trajectory parameters
    load_init = [1.0, 0, 0.78]
    init_pos_rel = [-1, 1, 0.8, 0]
    load_target_rel = [1.2, 0, 0]
    load_mass = 0.1

    pos_ref, vel_ref, yaw_ref, ctrl_type, _ = construct(init_pos_rel, load_target_rel, load_mass, plot_result=False)
    pos_ref = pos_ref + np.array([load_init])

    # Initialize the library
    if not glfw.init():
        return

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(960, 1080, "Crazyflie in MuJoCo", None, None)
    if not window:
        glfw.terminate()
        return

    # Make the window's context current
    glfw.make_context_current(window)

    # initialize visualization data structures
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation = 45, -10
    cam.lookat, cam.distance = [0, -0.6, 1], 3

    pert = mujoco.MjvPerturb()
    opt = mujoco.MjvOption()
    scn = mujoco.MjvScene(model, maxgeom=100)
    con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)

    ## To obtain inertia matrix
    mujoco.mj_step(model, data)
    ### Controller
    controller = RobustGeomControl(model, data, drone_type='large_quad')
    controller.delta_r = 0
    controller_lqr = PlanarLqrControl(model)
    mass = controller.mass

    L = 0.4
    simtime = 0.0
    simulation_step = 0.001
    control_step = 0.01
    graphics_step = 0.02

    q0 = np.roll(Rotation.from_euler('xyz', [0, 0, yaw_ref[0]]).as_quat(), 1)
    data.qpos[0:8] = np.hstack((pos_ref[0, :], q0, 0))
    episode_length = 10  # (pos_ref.shape[0] - 1) * control_step + 2.5

    logger = Logger(episode_length, control_step)
    start = time.time()
    num_imag = 0

    for i in range(int(episode_length / control_step)):
        # Get time and states
        simtime = data.time
        pos = data.qpos[0:3]
        quat = data.xquat[1, :]
        vel = data.qvel[0:3]
        ang_vel = data.sensordata[0:3]

        if simtime < 1:
            target_pos = pos_ref[0, :]
            target_rpy = np.array([0, 0, yaw_ref[0]])
            data.ctrl = controller.compute_pos_control(pos, quat, vel, ang_vel, target_pos, target_rpy=target_rpy)
        else:
            data.xfrc_applied[1, 0:3] = np.array([0.1, 0.1, 0])
            controller.delta_r = 0
            i_ = i - int(1/control_step)
            alpha = data.qpos[7]
            hook_pos = pos + L * np.array([np.sin(alpha), 0, -np.cos(alpha)])
            cur_load_pos = data.qpos[8:11] + np.array([0, 0, 0.3])
            if i_ < pos_ref.shape[0]:
                # Add the load mass to the feedforward term of geometric ctrl
                # if np.linalg.norm(hook_pos - cur_load_pos) < 0.25:
                #     # print('high mass')
                #     controller.mass = mass + 0.05
                # if np.linalg.norm(hook_pos - cur_load_pos) > 0.35:
                #     # print('low mass')
                #     controller.mass = mass
                target_pos = pos_ref[i_, :]
                target_vel = vel_ref[i_, :]
                target_rpy = np.array([0, 0, yaw_ref[i_]])
                if 'lqr' in ctrl_type[i_]:
                    controller.mass = mass + float(ctrl_type[i_][-5:])
                    controller_lqr.mass = mass + float(ctrl_type[i_][-5:])
                    data.ctrl = controller.compute_pos_control(pos, quat, vel, ang_vel, target_pos,
                                                               target_vel=target_vel, target_rpy=target_rpy)
                    alpha = data.qpos[7]
                    dalpha = data.qvel[6]
                    pos_ = pos.copy()
                    vel_ = vel.copy()
                    R_plane = np.array([[np.cos(yaw_ref[i_]), -np.sin(yaw_ref[i_])],
                                        [np.sin(yaw_ref[i_]), np.cos(yaw_ref[i_])]])
                    pos_[0:2] = R_plane.T @ pos_[0:2]
                    vel_[0:2] = R_plane.T @ vel_[0:2]
                    hook_pos = pos_ + L * np.array([-np.sin(alpha), 0, -np.cos(alpha)])
                    hook_vel = vel_ + L * dalpha * np.array([-np.cos(alpha), 0, np.sin(alpha)])
                    hook_pos = np.take(hook_pos, [0, 2])
                    hook_vel = np.take(hook_vel, [0, 2])
                    phi_Q = Rotation.from_quat(np.roll(quat, -1)).as_euler('xyz')[1]
                    dphi_Q = ang_vel[1]
                    target_pos_ = target_pos.copy()
                    target_pos_[0:2] = R_plane.T @ target_pos_[0:2]
                    target_pos_load = np.take(target_pos_, [0, 2]) - np.array([0, L])
                    lqr_ctrl = controller_lqr.compute_control(hook_pos,
                                                                hook_vel,
                                                                alpha,
                                                                dalpha,
                                                                phi_Q,
                                                                dphi_Q,
                                                                target_pos_load)
                    data.ctrl[0] = lqr_ctrl[0]
                    data.ctrl[2] = lqr_ctrl[2]
                elif 'geom_load' in ctrl_type[i_]:
                    controller.mass = mass + float(ctrl_type[i_][-5:])
                    data.ctrl = controller.compute_pos_control(pos, quat, vel, ang_vel, target_pos,
                                                               target_vel=target_vel, target_rpy=target_rpy)
                else:
                    controller.mass = mass
                    data.ctrl = controller.compute_pos_control(pos, quat, vel, ang_vel, target_pos,
                                                               target_vel=target_vel, target_rpy=target_rpy)
            else:
                target_pos = pos_ref[-1, :]
                target_vel = np.zeros(3)
                target_rpy = np.array([0, 0, yaw_ref[-1]])
                data.ctrl = controller.compute_pos_control(pos, quat, vel, ang_vel, target_pos,
                                                           target_vel=target_vel, target_rpy=target_rpy)
        for _ in range(int(control_step / simulation_step)):
            mujoco.mj_step(model, data, 1)
        state = np.hstack([target_pos - pos, Rotation.from_quat(quat).as_euler('xyz'), target_pos, pos, 0])
        logger.log(timestamp=simtime, state=state, control=data.ctrl)

        if i % (graphics_step / control_step) == 0:
            viewport = mujoco.MjrRect(0, 0, 0, 0)
            viewport.width, viewport.height = glfw.get_framebuffer_size(window)
            mujoco.mjv_updateScene(model, data, opt, pert=None, cam=cam, catmask=mujoco.mjtCatBit.mjCAT_ALL, scn=scn)
            mujoco.mjr_render(viewport, scn, con)

            glfw.swap_buffers(window)
            glfw.poll_events()

            # rgb = np.zeros((viewport.height, viewport.width, 3), dtype=np.uint8)
            # depth = np.zeros((viewport.height, viewport.width, 1))
            # mujoco.mjr_readPixels(rgb, depth, viewport=viewport, con=con)
            # rgb = np.flipud(rgb)
            # plt.imsave('../videos/grasp_nom/vid_'+str(num_imag)+'.png', rgb)
            # num_imag = num_imag + 1
            # sync with wall-clock time
            sync(i, start, control_step)

            if glfw.window_should_close(window):
                break
    # print('Load distance from target: x: ' + "{:.2f}".format(2.5 - cur_load_pos[0]) +
    #       ' m; y: ' + "{:.2f}".format(1.5 - cur_load_pos[1]) + ' m')
    glfw.terminate()
    logger.plot()
    # logger.plot3D()

if __name__ == '__main__':
    main()
