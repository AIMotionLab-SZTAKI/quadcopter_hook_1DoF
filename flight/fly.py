import mujoco
import glfw
import os
import numpy as np
from ctrl.GeomControl import GeomControl
import time
from assets.util import sync
from assets.logger import Logger


def main():
    # Reading model data
    print(f'Working directory:  {os.getcwd()}\n')
    model = mujoco.MjModel.from_xml_path("../assets/large_quad.xml")
    # use assets/cf2.xml for Crazyflie, but don't forget to change the ctrl gains, inertia and mass in GeomControl.py

    data = mujoco.MjData(model)

    # Initialize the library
    if not glfw.init():
        return

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(1280, 720, "Quadcopter simulation in MuJoCo", None, None)
    if not window:
        glfw.terminate()
        return

    # Make the window's context current
    glfw.make_context_current(window)

    # initialize visualization data structures
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation = 170, -30
    cam.lookat,  cam.distance  = [0, 0, 1], 3
    
    pert = mujoco.MjvPerturb()
    opt = mujoco.MjvOption()
    scn = mujoco.MjvScene(model, maxgeom=30)
    con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)

    ## To obtain inertia matrix
    mujoco.mj_step(model, data)
    ### Controller
    controller = GeomControl(model, data, drone_type='large_quad')

    """
    Optional stability analysis for the chosen ctrl gains
    
    c_1 = np.linspace(1e-7, 2, 100)
    c_2 = np.linspace(1e-7, 0.2, 100)

    for c1_ in c_1:
        for c2_ in c_2:
            print(controller.inertia)
            crit1, crit2, crit3 = controller.stability_analysis(controller.k_r, controller.k_v, controller.k_R,
                                                controller.k_w, c1_, c2_, controller.inertia, controller.mass)
            if crit1 and crit2 and crit3:
                print('Found params: ', c1_, c2_)
    """

    simtime = 0.0
    i = 0  # loop variable for syncing
    timestep = 0.005
    episode_length = 15
    logger = Logger(episode_length, timestep)
    start = time.time()

    while not glfw.window_should_close(window) and simtime < episode_length:
        simtime = data.time
        mujoco.mj_step(model, data, 1)
        viewport=mujoco.MjrRect(0,0,0,0)
        viewport.width, viewport.height = glfw.get_framebuffer_size(window)
        mujoco.mjv_updateScene(model, data, opt, pert=None, cam=cam, catmask=mujoco.mjtCatBit.mjCAT_ALL, scn=scn)
        mujoco.mjr_render(viewport, scn, con)

        if simtime < 1:
            target_pos = np.array([0, 0, 1])
            pos = data.qpos[0:3]
            quat = data.xquat[1, :]
            vel = data.qvel[0:3]
            ang_vel = data.sensordata[0:3]
            data.ctrl = controller.compute_pos_control(pos, quat, vel, ang_vel, target_pos)
        else:
            traj_freq = 1
            t = i * timestep
            target_pos = np.array([0.8 * np.sin(traj_freq * t), 0.8 * np.sin(2 * traj_freq * (t - np.pi / 2)),
                                   1])
            target_vel = np.array([0.8 * traj_freq * np.cos(traj_freq * t),
                                   0.8 * 2 * traj_freq * np.cos(2 * traj_freq * (t - np.pi / 2)), 0])
            pos = data.qpos[0:3]
            quat = data.xquat[1, :]
            vel = data.qvel[0:3]
            ang_vel = data.sensordata[0:3]
            target_yaw = 0*np.arctan2(target_vel[1], target_vel[0])
            target_rpy = np.array([0, 0, target_yaw])
            data.ctrl = controller.compute_pos_control(pos, quat, vel, ang_vel, target_pos, target_vel=target_vel,
                                                       target_rpy=target_rpy)

        glfw.swap_buffers(window)
        glfw.poll_events()

        state = np.hstack([target_pos-pos, np.zeros(10)])
        logger.log(timestamp=simtime, state=state, control=data.ctrl)

        # sync with wall-clock time
        sync(i, start, timestep)
        i = i+1

    logger.plot()
    glfw.terminate()


if __name__ == '__main__':
    main()
