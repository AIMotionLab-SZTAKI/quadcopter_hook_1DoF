from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import os


class Logger:
    """A class for logging and visualization.

    Stores, saves to file, and plots the kinematic information and RPMs
    of a simulation with one or more drones.

    """

    ################################################################################

    def __init__(self,
                 episode_length,
                 timestep,
                 num_runs=1,
                 sections=None
                 ):
        self.episode_steps = int(episode_length / timestep)
        self.timestamps = np.zeros((num_runs, self.episode_steps))
        #### Note: this is the suggest information to log ##############################
        self.states = np.zeros((num_runs, 13, self.episode_steps))  #### 13 states: pos_x,
        # pos_y,
        # pos_z,
        # vel_x,
        # vel_y,
        # vel_z,
        # quat_w,
        # quat_x,
        # quat_y,
        # quat_z,
        # ang_vel_x,
        # ang_vel_y,
        # ang_vel_z,

        #### Note: this is the suggest information to log ##############################
        self.controls = np.zeros((num_runs, 4, self.episode_steps))  #### 4 ctrl inputs
        self.current_step = 0
        self.sections = np.cumsum([1] + [elem for sublist in sections for elem in sublist]) if sections is not None else None

    ################################################################################

    def reset_counter(self):
        self.current_step = 0

    ################################################################################

    def log(self,
            timestamp,
            state,
            control=np.zeros(4),
            current_run=0,
            ):
        #### Log the information and increase the counter ##########
        if self.current_step > self.episode_steps:
            print('Somethings wrong in the logging department')
        else:
            self.timestamps[current_run, self.current_step] = timestamp
            self.states[current_run, :, self.current_step] = state
            self.controls[current_run, :, self.current_step] = control
            self.current_step = self.current_step + 1

    ################################################################################

    def save(self):
        """Save the logs to file.
        """
        with open(
                os.path.dirname(os.path.abspath(__file__)) + "/../../files/logs/save-flight-" + datetime.now().strftime(
                        "%m.%d.%Y_%H.%M.%S") + ".npy", 'wb') as out_file:
            np.savez(out_file, timestamps=self.timestamps, states=self.states, controls=self.controls)

    ################################################################################

    def save_as_csv(self,
                    comment: str = ""
                    ):
        # TODO
        csv_dir = os.environ.get('HOME') + "/Desktop/save-flight-" + comment + "-" + datetime.now().strftime(
            "%m.%d.%Y_%H.%M.%S")
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir + '/')
        # t = np.arange(0, self.timestamps.shape[1] / self.LOGGING_FREQ_HZ, 1 / self.LOGGING_FREQ_HZ)
        # with open(csv_dir + "/x" + str(i) + ".csv", 'wb') as out_file:
        #     np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 0, :]])), delimiter=",")
        # with open(csv_dir + "/y" + str(i) + ".csv", 'wb') as out_file:
        #     np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 1, :]])), delimiter=",")
        # with open(csv_dir + "/z" + str(i) + ".csv", 'wb') as out_file:
        #     np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 2, :]])), delimiter=",")
        # ####
        # with open(csv_dir + "/r" + str(i) + ".csv", 'wb') as out_file:
        #     np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 6, :]])), delimiter=",")
        # with open(csv_dir + "/p" + str(i) + ".csv", 'wb') as out_file:
        #     np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 7, :]])), delimiter=",")
        # with open(csv_dir + "/ya" + str(i) + ".csv", 'wb') as out_file:
        #     np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 8, :]])), delimiter=",")
        # ####
        # with open(csv_dir + "/rr" + str(i) + ".csv", 'wb') as out_file:
        #     rdot = np.hstack([0, (self.states[i, 6, 1:] - self.states[i, 6, 0:-1]) * self.LOGGING_FREQ_HZ])
        #     np.savetxt(out_file, np.transpose(np.vstack([t, rdot])), delimiter=",")
        # with open(csv_dir + "/pr" + str(i) + ".csv", 'wb') as out_file:
        #     pdot = np.hstack([0, (self.states[i, 7, 1:] - self.states[i, 7, 0:-1]) * self.LOGGING_FREQ_HZ])
        #     np.savetxt(out_file, np.transpose(np.vstack([t, pdot])), delimiter=",")
        # with open(csv_dir + "/yar" + str(i) + ".csv", 'wb') as out_file:
        #     ydot = np.hstack([0, (self.states[i, 8, 1:] - self.states[i, 8, 0:-1]) * self.LOGGING_FREQ_HZ])
        #     np.savetxt(out_file, np.transpose(np.vstack([t, ydot])), delimiter=",")
        # ###
        # with open(csv_dir + "/vx" + str(i) + ".csv", 'wb') as out_file:
        #     np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 3, :]])), delimiter=",")
        # with open(csv_dir + "/vy" + str(i) + ".csv", 'wb') as out_file:
        #     np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 4, :]])), delimiter=",")
        # with open(csv_dir + "/vz" + str(i) + ".csv", 'wb') as out_file:
        #     np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 5, :]])), delimiter=",")
        # ####
        # with open(csv_dir + "/wx" + str(i) + ".csv", 'wb') as out_file:
        #     np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 9, :]])), delimiter=",")
        # with open(csv_dir + "/wy" + str(i) + ".csv", 'wb') as out_file:
        #     np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 10, :]])), delimiter=",")
        # with open(csv_dir + "/wz" + str(i) + ".csv", 'wb') as out_file:
        #     np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 11, :]])), delimiter=",")
        # ####
        # with open(csv_dir + "/rpm0-" + str(i) + ".csv", 'wb') as out_file:
        #     np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 12, :]])), delimiter=",")
        # with open(csv_dir + "/rpm1-" + str(i) + ".csv", 'wb') as out_file:
        #     np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 13, :]])), delimiter=",")
        # with open(csv_dir + "/rpm2-" + str(i) + ".csv", 'wb') as out_file:
        #     np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 14, :]])), delimiter=",")
        # with open(csv_dir + "/rpm3-" + str(i) + ".csv", 'wb') as out_file:
        #     np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 15, :]])), delimiter=",")
            ####
    ################################################################################

    def plot(self):
        plt.rcParams.update({'font.size': 10, 'lines.linewidth': 1})
        fig, axs = plt.subplots(3, 1, figsize=(2.6, 4))
        t = self.timestamps[0, :]

        #### Column ################################################
        # col = 0

        #### XYZ ###################################################
        row = 0
        axs[row].plot(t, self.states[:, 0, :].T)
        axs[row].plot(t, self.states[:, 1, :].T)
        axs[row].plot(t, self.states[:, 2, :].T)
        axs[row].set_xlabel('time (s)')
        axs[row].set_ylabel('$e_r$ (m)')
        axs[row].legend(('$e_x$', '$e_y$', '$e_z$'))
        axs[row].set_ylim(-0.2, 0.12)
        axs[row].set_yticks((-0.2, -0.1, 0, 0.1))

        # row = 1
        # axs[row, col].plot(t, self.states[:, 1, :].T)
        # axs[row, col].set_xlabel('time (s)')
        # axs[row, col].set_ylabel('$e_y$ (m)')
        # # axs[row, col].set_ylim(-0.2, 0.08)
        #
        # row = 2
        # axs[row, col].plot(t, self.states[:, 2, :].T)
        # axs[row, col].set_xlabel('time (s)')
        # axs[row, col].set_ylabel('$e_z$ (m)')
        # # axs[row, col].set_ylim(-0.05, 0.1)

        # col = 1

        #### Controls ###############################################
        row = 1
        axs[row].plot(t, self.controls[:, 0, :].T)
        axs[row].set_xlabel('time (s)')
        axs[row].set_ylabel('F (N)')
        axs[row].set_ylim(5, 8)
        # axs[row].set_yticks((40, 42, 44, 46))

        row = 2
        axs[row].plot(t, self.controls[:, 1, :].T)
        axs[row].plot(t, self.controls[:, 2, :].T)
        axs[row].plot(t, self.controls[:, 3, :].T)
        axs[row].set_xlabel('time (s)')
        axs[row].set_ylabel(r'$\tau$ (Nm)')
        axs[row].legend((r'$\tau_x$', r'$\tau_y$', r'$\tau_z$'))
        axs[row].set_ylim(-0.3, 0.3)
        # axs[row].set_yticks((-1, -0.5, 0, 0.5, 1))

        # row = 2
        # axs[row, col].plot(t, self.controls[:, 2, :].T)
        # axs[row, col].set_xlabel('time (s)')
        # axs[row, col].set_ylabel(r'$\tau_y$ (Nm)')
        #
        # row = 3
        # axs[row, col].plot(t, self.controls[:, 3, :].T)
        # axs[row, col].set_xlabel('time (s)')
        # axs[row, col].set_ylabel(r'$\tau_z$ (Nm)')

        #### Drawing options #######################################
        for i in range(3):
            # for j in range(2):
            axs[i].grid(True)
            # axs[i, j].legend(loc='upper right',
            #                  frameon=True
            #                  )
        fig.subplots_adjust(left=0.271,
                            bottom=0.05,
                            right=0.99,
                            top=0.98,
                            wspace=0.5,
                            hspace=0.5
                            )
        plt.show(block=True)

    def plot3D(self):
        plt.rcParams.update({'font.size': 10, 'lines.linewidth': 1})
        e_r = self.states[0, 6:9, :] - self.states[0, 9:12, :]
        e_r_norm = np.linalg.norm(e_r, axis=0)
        print('RMSE:')
        print(np.sqrt(np.mean(e_r_norm**2)))
        print('max:')
        print(np.max(e_r_norm))
        print('max alpha:')
        print(np.max(np.abs(self.states[0, 12, :])))
        x = self.states[0, 9, :].tolist()
        y = self.states[0, 10, :].tolist()
        z = self.states[0, 11, :].tolist()
        xr = self.states[0, 6, :].tolist()
        yr = self.states[0, 7, :].tolist()
        zr = self.states[0, 8, :].tolist()
        points = np.array([xr, yr, zr]).T.reshape(-1, 1, 3)
        # segments = np.concatenate([points[:-1], points[1:]], axis=1)
        fig = plt.figure(figsize=(6, 4))
        ax = plt.axes(projection="3d")
        # Create a continuous norm to map from data points to colors

        # idx = {}
        # # idx['A'] = 0
        # idx['B'] = (np.linalg.norm(points[:, 0, 0:2] - np.array([-0.3, 0]), axis=1)).argmin()
        # idx['C'] = (np.linalg.norm(points[:, 0, 0:2], axis=1)).argmin()
        # idx['D'] = (np.linalg.norm(points[:, 0, :] - np.array([0.3, 0, 1.1]), axis=1)).argmin()
        # for (k, v) in idx.items():
        #     ax.scatter(xr[v], yr[v], zr[v], marker='x', color='black')
        #     ax.text(xr[v], yr[v], zr[v] + 0.1, k)

        # x, y, z, xr, yr, zr = x[idx['B']:], y[idx['B']:], z[idx['B']:], xr[idx['B']:], yr[idx['B']:], zr[idx['B']:]
        ax.plot3D(xr, yr, zr)
        ax.plot3D(x, y, z)
        ax.legend(('$r_d$ (m)', '$r$ (m)'))

        # traj_break_idx = np.argmax(np.abs(y) < 1e-4)
        # ax.scatter(x[traj_break_idx], y[traj_break_idx], z[traj_break_idx])
        # ax.scatter(0, 0, 0, marker='*')
        # ax.text(x[0], y[0], z[0], str([x[0], y[0], z[0]]))
        # ax.text(x[-1], y[-1], z[-1], "[{:.1f}, {:.1f}, {:.1f}]".format(x[-1], y[-1], z[-1]))
        ax.set_xlim(min(x) - 0.1, max(x) + 0.1)
        ax.set_ylim(min(y) - 0.1, max(y) + 0.1)
        ax.set_zlim(min(z) - 0.1, max(z) + 0.1)
        ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))
        ax.set_xlabel("$x$ (m)")
        ax.set_ylabel("$y$ (m)")
        ax.set_zlabel("$z$ (m)")
        plt.show(block=True)

    def plot2D(self):
        plt.rcParams.update({'font.size': 10, 'lines.linewidth': 1})
        fig, axs = plt.subplots(2, 2, figsize=(6, 2.66))
        t = self.timestamps[0, :]

        #### Column ################################################
        col = 0

        #### XYZ ###################################################
        row = 0
        axs[row, col].plot(t, self.states[:, 0, :].T)
        axs[row, col].plot(t, self.states[:, 1, :].T - self.states[:, 1, -1])
        axs[row, col].set_xlabel('time (s)')
        axs[row, col].set_ylabel('position (m)')
        axs[row, col].legend(('$x_L$', '$z_L$'))
        # axs[row, col].set_ylim(-0.2, 0.05)

        row = 1
        axs[row, col].plot(t, self.states[:, 2, :].T)
        axs[row, col].plot(t, self.states[:, 3, :].T)
        axs[row, col].set_xlabel('time (s)')
        axs[row, col].set_ylabel('angles (rad)')
        axs[row, col].legend((r'$\theta$', r'$\alpha$'))
        # axs[row, col].set_yticks([-0.25, 0, 0.25, 0.5])
        # axs[row, col].set_ylim(-0.2, 0.08)

        col = 1

        #### Controls ###############################################
        row = 0
        axs[row, col].plot(t, self.controls[:, 0, :].T)
        axs[row, col].set_xlabel('time (s)')
        axs[row, col].set_ylabel('F (N)')

        row = 1
        axs[row, col].plot(t, self.controls[:, 2, :].T)
        axs[row, col].set_xlabel('time (s)')
        axs[row, col].set_ylabel(r'$\tau$ (Nm)')

        #### Drawing options #######################################
        for i in range(2):
            for j in range(2):
                axs[i, j].grid(True)
                # axs[i, j].legend(loc='upper right',
                #                  frameon=True
                #                  )
        fig.subplots_adjust(left=0.132,
                            bottom=0.167,
                            right=0.98,
                            top=0.998,
                            wspace=0.316,
                            hspace=0.497
                            )
        plt.show(block=True)

    def plot_sequential(self):
        e_r = self.states[0, 6:9, :] - self.states[0, 9:12, :]
        e_r_norm = np.linalg.norm(e_r, axis=0)
        print('RMSE:')
        print(np.sqrt(np.mean(e_r_norm**2)))
        print('max:')
        print(np.max(e_r_norm))
        print('alpha max')
        print(np.max(np.abs(self.states[0, 12, :])))
        plt.rcParams.update({'font.size': 10, 'lines.linewidth': 1})
        fig, axs = plt.subplots(3, 2, figsize=(6, 4))
        t = self.timestamps[0, :]

        #### Column ################################################
        col = 0

        #### XYZ ###################################################
        row = 0
        axs[row, col].plot(t, self.states[:, 6, :].T)
        axs[row, col].plot(t, self.states[:, 9, :].T)

        lb = np.min(self.states[:, 9, :])
        ub = np.max(self.states[:, 9, :])
        if self.sections is not None:
            [axs[row, col].plot([t0, t0], [lb, ub], 'k-', linewidth=0.5) for t0 in self.sections]
            txt = 3 * ['A', 'B', 'C', 'D', 'E', 'F']
            [axs[row, col].text(t0-1.5, ub, txt_) for t0, txt_ in zip(self.sections, txt)]


        axs[row, col].set_xlabel('time (s)')
        axs[row, col].set_ylabel('$x$ (m)')
        axs[row, col].legend(('$x_d$', '$x$'), loc='upper right')
        axs[row, col].set_ylim(lb-0.1, ub+0.5)
        # axs[row].set_yticks((-0.2, -0.1, 0, 0.1))

        row = 1
        axs[row, col].plot(t, self.states[:, 7, :].T)
        axs[row, col].plot(t, self.states[:, 10, :].T)
        axs[row, col].set_xlabel('time (s)')
        axs[row, col].set_ylabel('$y$ (m)')
        axs[row, col].legend(('$y_d$', '$y$'), loc='upper right')
        # axs[row].set_ylim(-0.2, 0.12)
        # axs[row].set_yticks((-0.2, -0.1, 0, 0.1))

        row = 2
        axs[row, col].plot(t, self.states[:, 8, :].T)
        axs[row, col].plot(t, self.states[:, 11, :].T)
        axs[row, col].set_xlabel('time (s)')
        axs[row, col].set_ylabel('$z$ (m)')
        axs[row, col].legend(('$z_d$', '$z$'), loc='upper right')
        # axs[row].set_ylim(-0.2, 0.12)
        # axs[row].set_yticks((-0.2, -0.1, 0, 0.1))

        col = 1

        row = 0
        axs[row, col].plot(t, self.states[:, 12, :].T)
        axs[row, col].set_xlabel('time (s)')
        axs[row, col].set_ylabel(r'$\alpha$ (rad)')
        # axs[row].set_ylim(-0.2, 0.12)
        # axs[row].set_yticks((-0.2, -0.1, 0, 0.1))

        #### Controls ###############################################
        row = 1
        axs[row, col].plot(t[::10], self.controls[:, 0, ::10].T)
        axs[row, col].set_xlabel('time (s)')
        axs[row, col].set_ylabel('F (N)')
        # axs[row].set_ylim(5, 8)
        # axs[row].set_yticks((40, 42, 44, 46))

        row = 2
        # axs[row, col].plot(t[::10], self.controls[:, 1, ::10].T)
        # axs[row, col].plot(t[::10], self.controls[:, 2, ::10].T)
        # axs[row, col].plot(t[::10], self.controls[:, 3, ::10].T)
        axs[row, col].plot(t, self.states[:, 3, :].T)
        axs[row, col].plot(t, self.states[:, 4, :].T)
        axs[row, col].plot(t, np.abs(self.states[:, 5, :]).T)
        axs[row, col].set_ylabel('RPY (rad)')
        axs[row, col].legend((r'$\phi$', r'$\theta$', r'$\psi$'), loc='upper right')
        axs[row, col].set_xlabel('time (s)')
        # axs[row, col].set_ylabel(r'$\tau$ (Nm)')
        # axs[row, col].legend((r'$\tau_x$', r'$\tau_y$', r'$\tau_z$'), loc='upper right')
        # axs[row].set_ylim(-0.3, 0.3)
        # axs[row].set_yticks((-1, -0.5, 0, 0.5, 1))

        # row = 2
        # axs[row, col].plot(t, self.controls[:, 2, :].T)
        # axs[row, col].set_xlabel('time (s)')
        # axs[row, col].set_ylabel(r'$\tau_y$ (Nm)')
        #
        # row = 3
        # axs[row, col].plot(t, self.controls[:, 3, :].T)
        # axs[row, col].set_xlabel('time (s)')
        # axs[row, col].set_ylabel(r'$\tau_z$ (Nm)')

        #### Drawing options #######################################
        for i in range(3):
            for j in range(2):
                axs[i, j].grid(True)
                # axs[i, j].legend(loc='upper right',
                #                  frameon=True
                #                  )
        fig.subplots_adjust(left=0.084,
                            bottom=0.11,
                            right=0.98,
                            top=0.998,
                            wspace=0.287,
                            hspace=0.497
                            )
        plt.show(block=True)

    def plot_sequential_v2(self):
        plt.rcParams.update({'font.size': 8, 'lines.linewidth': 1})
        fig, axs = plt.subplots(6, 1, figsize=(3, 6))
        t = self.timestamps[0, :]

        #### Column ################################################
        col = 0

        #### XYZ ###################################################
        row = 0
        axs[row].plot(t, self.states[:, 6, :].T)
        axs[row].plot(t, self.states[:, 9, :].T)

        lb = np.min(self.states[:, 9, :])
        ub = np.max(self.states[:, 9, :])
        [axs[row].plot([t0, t0], [lb, ub], 'k-', linewidth=0.5) for t0 in self.sections]
        txt = 3 * ['A', 'B', 'C', 'D', 'E']
        [axs[row].text(t0 - 1, ub, txt_) for t0, txt_ in zip(self.sections, txt)]

        axs[row].set_xlabel('time (s)')
        axs[row].set_ylabel('$x$ (m)')
        axs[row].legend(('$x_d$', '$x$'), loc='upper right')
        axs[row].set_ylim(lb - 0.1, ub + 0.7)
        # axs[row].set_yticks((-0.2, -0.1, 0, 0.1))

        row = 1
        axs[row].plot(t, self.states[:, 7, :].T)
        axs[row].plot(t, self.states[:, 10, :].T)
        axs[row].set_xlabel('time (s)')
        axs[row].set_ylabel('$y$ (m)')
        axs[row].legend(('$y_d$', '$y$'), loc='upper right')
        # axs[row].set_ylim(-0.2, 0.12)
        # axs[row].set_yticks((-0.2, -0.1, 0, 0.1))
        [axs[row].plot([t0, t0], [lb, ub], 'k-', linewidth=0.5) for t0 in self.sections]

        row = 2
        axs[row].plot(t, self.states[:, 8, :].T)
        axs[row].plot(t, self.states[:, 11, :].T)
        axs[row].set_xlabel('time (s)')
        axs[row].set_ylabel('$z$ (m)')
        axs[row].legend(('$z_d$', '$z$'), loc='upper right')
        # axs[row].set_ylim(-0.2, 0.12)
        # axs[row].set_yticks((-0.2, -0.1, 0, 0.1))
        [axs[row].plot([t0, t0], [lb, ub], 'k-', linewidth=0.5) for t0 in self.sections]

        col = 1

        row = 3
        axs[row].plot(t, self.states[:, 12, :].T)
        axs[row].set_xlabel('time (s)')
        axs[row].set_ylabel(r'$\alpha$ (rad)')
        # axs[row].set_ylim(-0.2, 0.12)
        # axs[row].set_yticks((-0.2, -0.1, 0, 0.1))
        [axs[row].plot([t0, t0], [lb, ub], 'k-', linewidth=0.5) for t0 in self.sections]

        #### Controls ###############################################
        row = 4
        axs[row].plot(t, self.controls[:, 0, :].T)
        axs[row].set_xlabel('time (s)')
        axs[row].set_ylabel('F (N)')
        # axs[row].set_ylim(5, 8)
        # axs[row].set_yticks((40, 42, 44, 46))
        [axs[row].plot([t0, t0], [lb, ub], 'k-', linewidth=0.5) for t0 in self.sections]

        row = 5
        axs[row].plot(t, self.controls[:, 1, :].T)
        axs[row].plot(t, self.controls[:, 2, :].T)
        axs[row].plot(t, self.controls[:, 3, :].T)
        axs[row].set_xlabel('time (s)')
        axs[row].set_ylabel(r'$\tau$ (Nm)')
        axs[row].legend((r'$\tau_x$', r'$\tau_y$', r'$\tau_z$'), loc='upper right')
        # axs[row].set_ylim(-0.3, 0.3)
        # axs[row].set_yticks((-1, -0.5, 0, 0.5, 1))
        [axs[row].plot([t0, t0], [lb, ub], 'k-', linewidth=0.5) for t0 in self.sections]

        # row = 2
        # axs[row, col].plot(t, self.controls[:, 2, :].T)
        # axs[row, col].set_xlabel('time (s)')
        # axs[row, col].set_ylabel(r'$\tau_y$ (Nm)')
        #
        # row = 3
        # axs[row, col].plot(t, self.controls[:, 3, :].T)
        # axs[row, col].set_xlabel('time (s)')
        # axs[row, col].set_ylabel(r'$\tau_z$ (Nm)')

        #### Drawing options #######################################
        for i in range(6):
            # for j in range(2):
            axs[i].grid(False)
        fig.subplots_adjust(left=0.16,
                            bottom=0.08,
                            right=0.98,
                            top=0.998,
                            wspace=0.287,
                            hspace=0.733
                            )
        plt.show(block=True)
