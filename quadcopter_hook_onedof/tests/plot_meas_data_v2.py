import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def import_usd_log(filename):
    with open(filename, 'rb') as f:
        meas = pickle.load(f)
    return meas


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


# Create a function to handle mouse clicks
def onclick_4_points(event):
    global idx
    if event.button == 1:
        idx.append(int(event.xdata))
    if len(idx) == 4:
        fig.canvas.mpl_disconnect(cid)
        plt.close()


def onclick_6_points(event):
    global idx
    if event.button == 1:
        idx.append(int(event.xdata))
    if len(idx) == 6:
        fig.canvas.mpl_disconnect(cid)
        plt.close()


def onclick_1_point(event):
    global idx
    idx = int(event.xdata)
    fig.canvas.mpl_disconnect(cid)
    plt.close()


def plot_meas_res(t, r, rd, rpy, alpha, ctrl, idx):
    plt.rcParams.update({'font.size': 10, 'lines.linewidth': 1.5})
    e_r = r - rd
    e_r_norm = np.linalg.norm(e_r, axis=1)
    print('RMSE:')
    print(np.sqrt(np.mean(e_r_norm ** 2)))
    print('max:')
    print(np.max(e_r_norm))
    print('max alpha:')
    print(np.max(np.abs(alpha)))
    print('error at detaching:')
    print(np.linalg.norm(e_r[idx[4], 0:2]))

    grasp_col = 'k--'
    detach_col = 'k--'
    vert_width = 1

    ################## Plot 1: reference pos in 3D  ############################

    points = np.expand_dims(rd, axis=1)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    fig = plt.figure(figsize=(4, 4))
    ax = plt.axes(projection="3d")
    # # Create a continuous norm to map from data points to colors
    vel_dim = np.gradient(rd, 0.01, axis=0)
    vel = np.linalg.norm(vel_dim, axis=1)
    vel_av = np.hstack((moving_average(vel, 3), np.zeros(2)))
    norm = plt.Normalize(0, np.max(vel_av))
    lc = Line3DCollection(segments, cmap='jet', norm=norm)
    # Set the values used for colormapping
    lc.set_array(vel_av)
    lc.set_linewidth(2)
    line = ax.add_collection(lc)
    cbar = fig.colorbar(line, location='top', shrink=0.7, pad=0.05)
    cbar.set_label("velocity (m/s)")

    # ax.plot3D(2*[rd[idx0, 0]], 2*[rd[idx0, 1]], [np.min(rd[:, 2]), np.max(rd[:, 2])], grasp_col)
    # ax.plot3D(2*[rd[idx1, 0]], 2*[rd[idx1, 1]], [np.min(rd[:, 2]), np.max(rd[:, 2])], detach_col)

    for i, num in enumerate(idx):
        ax.scatter(rd[num, 0], rd[num, 1], rd[num, 2], marker='x', color='black')
        ax.text(rd[num, 0], rd[num, 1], rd[num, 2] + 0.2, chr(65+i))

    ax.set_xlim(min(r[:, 0]) - 0.3, max(r[:, 0]) + 0.3)
    ax.set_ylim(min(r[:, 1]) - 0.3, max(r[:, 1]) + 0.3)
    ax.set_zlim(min(r[:, 2]) - 0.3, max(r[:, 2]) + 0.4)
    ax.set_xticks(np.arange(np.ceil(2*(min(r[:, 0]) - 0.3))/2, np.ceil(2*(max(r[:, 0]) + 0.3))/2, 0.5))
    ax.set_yticks(np.arange(np.ceil(2*(min(r[:, 1]) - 0.3))/2, np.ceil(2*(max(r[:, 1]) + 0.3))/2, 0.5))
    ax.set_zticks(np.arange(np.ceil(2*(min(r[:, 2]) - 0.3))/2, np.ceil(2*(max(r[:, 2]) + 0.4))/2, 0.5))
    ax.set_box_aspect((np.ptp(r[:, 0]), np.ptp(r[:, 1]), np.ptp(r[:, 2])))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    fig.subplots_adjust(left=0.086,
                        bottom=0.052,
                        right=0.832,
                        top=0.854
                        )

    ##################### Plot 2 #############################
    fig, axs = plt.subplots(5, 1, figsize=(2.6, 8))
    t0 = t[idx[2]]
    t1 = t[idx[4]]

    row = 0
    axs[row].plot(t, r-rd)
    lb = np.min(r - rd)
    ub = np.max(r - rd)
    for i in idx:
        axs[row].plot([t[i], t[i]], [lb, ub], grasp_col, linewidth=vert_width)
        # axs[row].plot([t1, t1], [lb, ub], detach_col, linewidth=vert_width)
    axs[row].set_xlabel('time (s)')
    axs[row].set_ylabel('$e_r$ (m)')
    axs[row].legend(('$e_x$', '$e_y$', '$e_z$'), ncol=3, loc="upper right")
    axs[row].grid(True)
    # axs[row].set_ylim(-0.2, 0.12)
    axs[row].set_xticks(np.arange(0, np.max(t), 5))

    row = 1
    angles = np.hstack((rpy[:, 0:2], np.expand_dims(alpha, 1)))
    axs[row].plot(t, angles)
    axs[row].set_xlabel('time (s)')
    axs[row].set_ylabel('angles (rad)')
    axs[row].legend(('$\phi$', r'$\theta$', r'$\alpha$'), ncol=3, loc="upper right")
    axs[row].grid(True)
    lb = np.min(angles)
    ub = np.max(angles)
    for i in idx:
        axs[row].plot([t[i], t[i]], [lb, ub], grasp_col, linewidth=vert_width)
        # axs[row].plot([t1, t1], [lb, ub], detach_col, linewidth=vert_width)
    # axs[row].set_ylim(5, 8)
    axs[row].set_xticks(np.arange(0, np.max(t), 5))

    row = 2
    # prop_cycle = plt.rcParams['axes.prop_cycle']
    # colors = prop_cycle.by_key()['color']
    axs[row].plot(t, rpy[:, 2],)
    axs[row].plot(t, rpy[:, 3], dashes=[4, 4])
    axs[row].set_xlabel('time (s)')
    axs[row].set_ylabel(r'$\psi$ (rad)')
    axs[row].legend(('$\psi$', r'$\psi_d$'), ncol=2, loc="upper right")
    axs[row].grid(True)
    lb = np.min(rpy[:, 2])
    ub = np.max(rpy[:, 2])
    for i in idx:
        axs[row].plot([t[i], t[i]], [lb, ub], grasp_col, linewidth=vert_width)
        # axs[row].plot([t1, t1], [lb, ub], detach_col, linewidth=vert_width)
    axs[row].set_xticks(np.arange(0, np.max(t), 5))

    row = 3
    F_av = moving_average(ctrl[:, 0], 10)
    axs[row].plot(t, np.hstack((F_av, F_av[-1]*np.ones(9))))
    axs[row].set_xlabel('time (s)')
    axs[row].set_ylabel('$F$ (N)')
    axs[row].grid(True)
    lb = np.min(ctrl[:, 0])
    ub = np.max(ctrl[:, 0])
    for i in idx:
        axs[row].plot([t[i], t[i]], [lb, ub], grasp_col, linewidth=vert_width)
        # axs[row].plot([t1, t1], [lb, ub], detach_col, linewidth=vert_width)
    axs[row].set_xticks(np.arange(0, np.max(t), 5))

    row = 4
    tau = np.asarray([np.hstack((moving_average(x, 10), np.zeros(9))) for x in ctrl[:, 1:].T]).T
    axs[row].plot(t, tau)
    axs[row].set_xlabel('time (s)')
    axs[row].set_ylabel(r'$\tau$ (Nm)')
    axs[row].grid(True)
    axs[row].legend((r'$\tau_x$', r'$\tau_y$', r'$\tau_z$'), ncol=3, loc="upper right")
    # axs[row].set_ylim(-0.3, 0.3)
    lb = np.min(ctrl[:, 1:])
    ub = np.max(ctrl[:, 1:])
    for i in idx:
        axs[row].plot([t[i], t[i]], [lb, ub], grasp_col, linewidth=vert_width)
        # axs[row].plot([t1, t1], [lb, ub], detach_col, linewidth=vert_width)
    axs[row].set_xticks(np.arange(0, np.max(t), 5))

    fig.subplots_adjust(left=0.236,
                        bottom=0.074,
                        right=0.98,
                        top=0.998,
                        wspace=0.287,
                        hspace=0.497
                        )

    plt.show()




if __name__ == '__main__':
    meas = True
    if meas:
        # Process micro SD card data
        usd_raw = import_usd_log('data/hook_flight_scen_3.pickle')
        usd_freq = 100

        # Create the plot
        fig, ax = plt.subplots()
        ax.plot(usd_raw["ctrltargetZ.x"])
        ax.plot(usd_raw["ctrltargetZ.y"])
        ax.plot(usd_raw["ctrltargetZ.z"])

        # Register the function as a callback for mouse clicks
        idx = []
        cid = fig.canvas.mpl_connect('button_press_event', onclick_6_points)
        plt.show()
        waypoints_idx = idx.copy()

        start = waypoints_idx[0]
        end = waypoints_idx[-1]
        episode_length_sec = (end - start) / usd_freq

        waypoints_idx = [i - start for i in waypoints_idx]
        waypoints_idx[-1] -= 1

        data = dict()

        data['t'] = (usd_raw["timestamp"][start:end] - usd_raw["timestamp"][start]) / 1000
        data['x'] = usd_raw["stateEstimateZ.x"][start:end]/1000
        data['y'] = usd_raw["stateEstimateZ.y"][start:end]/1000
        data['z'] = usd_raw["stateEstimateZ.z"][start:end]/1000
        data['xr'] = usd_raw["ctrltargetZ.x"][start:end]/1000
        data['yr'] = usd_raw["ctrltargetZ.y"][start:end]/1000
        data['zr'] = usd_raw["ctrltargetZ.z"][start:end]/1000
        data['roll'] = usd_raw["stateEstimate.roll"][start:end]*np.pi/180
        data['pitch'] = usd_raw["stateEstimate.pitch"][start:end]*np.pi/180
        data['yaw'] = usd_raw["stateEstimate.yaw"][start:end]*np.pi/180
        data['Mx'] = usd_raw["controller.ctr_roll"][start:end]
        data['My'] = usd_raw["controller.ctr_pitch"][start:end]
        data['Mz'] = usd_raw["controller.ctr_yaw"][start:end]
        data['F'] = usd_raw["controller.ctr_thrust"][start:end]*1.12
        data['alpha'] = usd_raw["load_pose.alpha"][start:end]
        data['yawr'] = usd_raw["controller.target_yaw"][start:end]*np.pi/180
    else:
        data = import_usd_log('data/simu_test.pickle')
        start = 0
        # Create the plot
        fig, ax = plt.subplots()
        ax.plot(data['xr'])
        ax.plot(data['yr'])
        ax.plot(data['zr'])

        # Register the function as a callback for mouse clicks
        idx = []
        cid = fig.canvas.mpl_connect('button_press_event', onclick_4_points)
        plt.show()
        simu_idx = idx.copy()
        waypoints_idx = [0] + simu_idx + [data['t'].shape[0]-1]


    # Construct states
    t = data['t']
    pos = np.vstack((data['x'], data['y'], data['z'])).T
    target_pos = np.vstack((data['xr'], data['yr'], data['zr'])).T
    ctrl = np.vstack((data['F'], data['Mx'], data['My'], data['Mz'])).T

    rpy = np.vstack((data['roll'], data['pitch'], data['yaw'], data['yawr'])).T
    alpha = data['alpha']

    plot_meas_res(t, pos, target_pos, rpy, alpha, ctrl, waypoints_idx)
