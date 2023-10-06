from setuptools import setup, find_packages

setup(name='quadcopter_hook_onedof',
      version='1.0.0',
      packages=find_packages(),
      install_requires=[
        'numpy==1.22.3',
        'mujoco==2.1.5',
        'glfw==2.5.3',
        'scipy==1.8.0',
        'matplotlib==3.5.2',
        'sympy==1.10.1',
        'mosek==9.3.21',
        'control==0.9.2',
        'casadi==3.6.3',
        'cvxopt @ https://github.com/AIMotionLab-SZTAKI/cvxopt/raw/mosek_handler/dist/cvxopt-0+untagged.56.g7e7c97b.dirty-cp310-cp310-linux_x86_64.whl'
        ]
      )
