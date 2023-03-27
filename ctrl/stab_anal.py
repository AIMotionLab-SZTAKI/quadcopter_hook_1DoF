import mujoco
import os
import numpy as np
from ctrl.GeomControl import GeomControl
from ctrl.RobustGeomControl import RobustGeomControl


if __name__ == '__main__':
    # Reading model data
    print(f'Working directory:  {os.getcwd()}\n')
    model = mujoco.MjModel.from_xml_path("../hook_up_scenario/hook_scenario.xml")
    data = mujoco.MjData(model)

    ## To obtain inertia matrix
    mujoco.mj_step(model, data)
    ### Controller
    controller = RobustGeomControl(model, data, drone_type='large_quad')

    c_1 = np.linspace(5, 20, 400)
    c_2 = np.linspace(5, 20, 400)
    eps = [1e-6, 1e-6]

    for c1_ in c_1:
        for c2_ in c_2:
            crit1, crit2, crit3, crit4 = controller.stability_analysis(controller.k_r, controller.k_v, controller.k_R,
                                                controller.k_w, c1_, c2_, np.diag(controller.inertia), controller.mass, eps)
            if crit1 and crit2 and crit3 and crit4:
                print('Found params: ', c1_, c2_)
