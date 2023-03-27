import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    n = 6
    d = 0.06
    L = d / 2 * np.sin(np.pi / n)
    R = L / np.tan(np.pi / n)
    num_sec = int(np.floor(3/4*n))
    p = [[R * np.sin((2 * i + 1) * np.pi / n), -R * np.cos((2 * i + 1) * np.pi / n) + d/2 + 0.04] for i in range(num_sec)]
    phi = [-np.pi / 2 + (2 * i + 1) * np.pi / n for i in range(num_sec)]
    # plt.figure()
    # [plt.plot(p_[0], p_[1], 'o') for p_ in p]
    # plt.show()
    with open('../assets/hook_scenario.xml', 'r') as file:
        # read a list of lines into data
        lines = file.readlines()

    L = 1.2 * L
    for i in range(num_sec):
        lines[35 + i] = '            <geom type="capsule" pos="0 ' + '{:.5f}'.format(p[i][0]) + \
                        ' ' + '{:.5f}'.format(p[i][1]) + '" euler="' + '{:.5f}'.format(phi[i]) + \
                        ' 0 0" size="0.0035 ' + '{:.5f}'.format(L) + \
                        '" mass="0.0001"/>\n'

    n = 8
    d = 0.06
    L = d / 2 * np.sin(np.pi / n)
    R = L / np.tan(np.pi / n)
    num_sec = int(np.floor(3/4*n))
    p = [[R * np.sin((2 * i + 1) * np.pi / n), -R * np.cos((2 * i + 1) * np.pi / n) + d/2 + 0.04] for i in range(num_sec)]
    phi = [-np.pi / 2 + (2 * i + 1) * np.pi / n for i in range(num_sec)]
    L = 1.2 * L
    for i in range(num_sec):
        lines[64 + i] = '            <geom type="capsule" pos="0 ' + '{:.5f}'.format(p[i][0]) + \
                        ' ' + '{:.5f}'.format(p[i][1]) + '" euler="' + '{:.5f}'.format(phi[i]) + \
                        ' 0 0" size="0.005 ' + '{:.5f}'.format(L) + \
                        '" mass="0.0001"/>\n'
    # and write everything back
    with open('../assets/hook_scenario.xml', 'w') as file:
        file.writelines(lines)
