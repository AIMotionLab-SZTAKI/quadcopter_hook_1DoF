import pickle
import mujoco
import glfw
import numpy as np
from quadcopter_hook_onedof.ctrl.RobustGeomControl import RobustGeomControl
from quadcopter_hook_onedof.ctrl.LqrControl import LqrControl
import time
from quadcopter_hook_onedof.assets.util import sync
from scipy.spatial.transform import Rotation
from quadcopter_hook_onedof.assets.logger import Logger
from quadcopter_hook_onedof.planning.traj_opt_min_time import construct, plot_3d_trajectory


def main():
    # Reading model data
    model = mujoco.MjModel.from_xml_path("../assets/hook_scenario.xml")
    data = mujoco.MjData(model)

    scenario_data = [[np.array([-0.9, 1.1, 1.1]), np.deg2rad(0), np.array([0.7, 1.1, 0.64]), np.deg2rad(-55),
                     np.array([1.0, -1.1, 0.66]), np.deg2rad(0), 0.05],
                     [np.array([-0.9, 1.1, 1.1]), np.deg2rad(0), np.array([0., 0., 0.64]), np.deg2rad(-35),
                     np.array([1.0, -1.1, 0.66]), np.deg2rad(-90), 0.05],
                     [np.array([-0.9, 1.1, 1.1]), np.deg2rad(0), np.array([0.8, -1.0, 0.64]), np.deg2rad(30),
                     np.array([0.3, 1.1, 0.66]), np.deg2rad(0), 0.075],
                     [np.array([-0.9, 1.1, 1.1]), np.deg2rad(0), np.array([0.9, 0.8, 0.64]), np.deg2rad(-120),
                     np.array([-0.9, -1.1, 0.66]), np.deg2rad(185), 0.075]]

    scen_num = 3
    # Input: payload initial position, yaw + drone initial position, yaw + payload final position
    init_pos_drone = scenario_data[scen_num][0]  # Drone position after takeoff
    init_yaw_drone = scenario_data[scen_num][1]  # Drone yaw at takeoff
    init_pos_load = scenario_data[scen_num][2]  # Load initial position, only z coordinate is offset
    init_yaw_load = scenario_data[scen_num][3]  # Load initial orientation
    target_pos_load = scenario_data[scen_num][4]  # Load target position
    target_yaw_load = scenario_data[scen_num][5]  # Load target yaw
    load_mass = scenario_data[scen_num][6]
    model.body_mass[4] = load_mass

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

    pos_ref, vel_ref, yaw_ref, ctrl_type, T = construct(init_pos=init_pos_drone_rel, load_target=target_pos_load_rel,
                                                        load_mass=load_mass, plot_result=False, save_splines=False,
                                                        save_path=None, init_pos_abs=init_pos_load,
                                                        load_init_yaw=init_yaw_load, load_target_yaw=target_yaw_load)

    # Initialize the library
    if not glfw.init():
        return

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(1920, 1080, "Payload transportation in MuJoCo", None, None)
    if not window:
        glfw.terminate()
        return

    # Make the window's context current
    glfw.make_context_current(window)

    # initialize visualization data structures
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation = 120, -25
    cam.lookat, cam.distance = [1., -3, 1.7], 1.3

    opt = mujoco.MjvOption()
    scn = mujoco.MjvScene(model, maxgeom=100)
    con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)

    ## To obtain inertia matrix
    mujoco.mj_step(model, data)
    ### Controller
    controller = RobustGeomControl(model, data, drone_type='large_quad')
    # controller.delta_r = 0
    mass = controller.mass
    controller_lqr = LqrControl(model)

    L = 0.4
    simtime = 0.0
    simulation_step = 0.001
    control_step = 0.01
    graphics_step = 0.02
    num_imag = 0

    q0 = np.roll(Rotation.from_euler('xyz', [0, 0, yaw_ref[0]]).as_quat(), 1)
    data.qpos[0:8] = np.hstack((pos_ref[0, :], q0, 0))
    data.qpos[8:] = np.hstack((init_pos_load[0:2], 0,
                               np.roll(Rotation.from_euler('xyz', [0, 0, init_yaw_load]).as_quat(), 1)))
    episode_length = (pos_ref.shape[0] - 1) * control_step

    logger = Logger(episode_length, control_step)
    start = time.time()

    for i in range(int(episode_length / control_step)):
        # Get time and states
        simtime = data.time
        pos = data.qpos[0:3]
        quat = data.xquat[1, :]
        vel = data.qvel[0:3]
        ang_vel = data.sensordata[0:3]
        alpha = data.qpos[7]
        if simtime < 1:
            target_pos = pos_ref[0, :]
            target_rpy = np.array([0, 0, yaw_ref[0]])
            data.ctrl = controller.compute_pos_control(pos, quat, vel, ang_vel, target_pos, target_rpy=target_rpy)
        else:

            i_ = i - int(1/control_step)

            if i_ < pos_ref.shape[0]:
                target_pos = pos_ref[i_, :]
                target_vel = vel_ref[i_, :]
                target_rpy = np.array([0, 0, yaw_ref[i_]])
                if 'lqr' in ctrl_type[i_]:
                    controller.mass = mass + float(ctrl_type[i_][-5:])
                    controller_lqr.mass = mass + float(ctrl_type[i_][-5:])
                    dalpha = data.qvel[6]
                    data.ctrl = controller_lqr.compute_control(pos,
                                                               quat,
                                                               vel,
                                                               ang_vel,
                                                               alpha,
                                                               dalpha,
                                                               target_pos,
                                                               target_rpy=target_rpy)
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

        load_pos_3d = pos + Rotation.from_quat(quat).as_matrix() @ np.array([-np.sin(alpha), 0, np.cos(alpha)]) * L
        state = np.hstack([pos, Rotation.from_quat(np.roll(quat, -1)).as_euler('xyz'), target_pos, pos, alpha])

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
            # plt.imsave('../videos/transport_3_cubes/vid_'+str(num_imag)+'.png', rgb)
            # num_imag = num_imag + 1
            # sync with wall-clock time
            sync(i, start, control_step)

            if glfw.window_should_close(window):
                break

    glfw.terminate()
    logger.plot_sequential()
    logger.plot3D()

    simu = dict()
    start = int(1/control_step)
    simu['t'] = logger.timestamps[0, start:]
    simu['x'] = logger.states[0, 0, start:]
    simu['y'] = logger.states[0, 1, start:]
    simu['z'] = logger.states[0, 2, start:]
    simu['xr'] = logger.states[0, 6, start:]
    simu['yr'] = logger.states[0, 7, start:]
    simu['zr'] = logger.states[0, 8, start:]
    simu['roll'] = logger.states[0, 3, start:]
    simu['pitch'] = logger.states[0, 4, start:]
    simu['yaw'] = logger.states[0, 5, start:]
    simu['Mx'] = logger.controls[0, 1, start:]
    simu['My'] = logger.controls[0, 2, start:]
    simu['Mz'] = logger.controls[0, 3, start:]
    simu['F'] = logger.controls[0, 0, start:]
    simu['alpha'] = logger.states[0, 12, start:]

    # with open('data/simu_test.pickle', 'wb') as file:
    #     pickle.dump(simu, file)


if __name__ == '__main__':
    main()
