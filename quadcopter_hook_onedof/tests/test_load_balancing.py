import mujoco
import glfw
import numpy as np
from quadcopter_hook_onedof.ctrl.LqrControl import LqrControl
import time
from quadcopter_hook_onedof.assets.util import sync
from scipy.spatial.transform import Rotation
from quadcopter_hook_onedof.assets.logger import Logger

def main():
    # Reading model data
    model = mujoco.MjModel.from_xml_path("../assets/hook_scenario.xml")
    data = mujoco.MjData(model)

    # Initialize the library
    if not glfw.init():
        return

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(1920, 1080, "Crazyflie in MuJoCo", None, None)
    if not window:
        glfw.terminate()
        return

    # Make the window's context current
    glfw.make_context_current(window)

    # initialize visualization data structures
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation = 60, -30
    cam.lookat, cam.distance = [0, 0, 0.6], 2

    pert = mujoco.MjvPerturb()
    opt = mujoco.MjvOption()
    scn = mujoco.MjvScene(model, maxgeom=100)
    con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)

    ## To obtain inertia matrix
    mujoco.mj_step(model, data)
    ### Controller
    controller = LqrControl(model)

    L = 0.4
    simtime = 0.0
    simulation_step = 0.001
    control_step = 0.01
    graphics_step = 0.02

    target_pos = np.array([0, 0, 1])
    target_pos_load = target_pos - np.array([0, 0, L])
    alpha = np.pi/10
    yaw = 0
    R_plane = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    init_load_pos = target_pos + L * np.array([-np.sin(alpha) - 0.05, 0, -np.cos(alpha) - 0.55])
    init_load_pos[0:2] = R_plane @ init_load_pos[0:2]
    target_pos_load[0:2] = R_plane @ target_pos_load[0:2]

    data.qpos[0:3] = target_pos #+ np.array([0.1, 0, 0])

    q0 = np.roll(Rotation.from_euler('ZYX', [yaw, 0, 0]).as_quat(), 1)
    data.qpos[3:7] = q0
    data.qpos[7] = alpha
    data.qpos[8:11] = init_load_pos
    data.qpos[11:15] = q0
    episode_length = 10

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
        alpha = data.qpos[7]
        dalpha = data.qvel[6]

        if 3 < simtime < 3.1:
            data.xfrc_applied[4, 0:2] = 0.3
        else:
            data.xfrc_applied = 0

        data.ctrl = controller.compute_control(pos,
                                               quat,
                                               vel,
                                               ang_vel,
                                               alpha,
                                               dalpha,
                                               target_pos)
        for _ in range(int(control_step / simulation_step)):
            mujoco.mj_step(model, data, 1)
        state = np.hstack([pos, alpha, np.zeros(9)])
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
            # plt.imsave('../videos/lqr_stab/vid_'+str(num_imag)+'.png', rgb)
            # num_imag += 1
            # sync with wall-clock time
            sync(i, start, control_step)

            if glfw.window_should_close(window):
                break
    # print('Load distance from target: x: ' + "{:.2f}".format(2.5 - cur_load_pos[0]) +
    #       ' m; y: ' + "{:.2f}".format(1.5 - cur_load_pos[1]) + ' m')
    glfw.terminate()
    logger.plot2D()
    # mujoco.mjv_applyPerturbForce()

if __name__ == '__main__':
    main()
