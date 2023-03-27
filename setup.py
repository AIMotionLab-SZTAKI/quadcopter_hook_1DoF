from setuptools import setup

setup(name='quadcopter_hook_1DoF',
      version='1.0.0',
      install_requires=[
        'numpy==1.22.3',
        'mujoco==2.1.5',
        'glfw==2.5.3',
        'scipy==1.8.0',
        'matplotlib==3.5.2',
        'sympy==1.10.1',
        'mosek==9.3.21',
        'control==0.9.2',
        'cvxopt @ https://github.com/AIMotionLab-SZTAKI/cvxopt/raw/mosek_handler/dist/cvxopt-0%2Buntagged.55.gc611b51.dirty-cp38-cp38-linux_x86_64.whl'
        ]
      )
