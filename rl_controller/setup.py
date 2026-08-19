from glob import glob
import os

from setuptools import find_packages, setup


package_name = "rl_controller"
model = "2026-08-18_12-18-31_full_seed42/exported/policy.onnx"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "models"), [model]),
    ],
    install_requires=["setuptools", "numpy", "onnxruntime"],
    zip_safe=True,
    maintainer="PiBot maintainers",
    maintainer_email="rosdev@todo.todo",
    description="ROS 2 deployment node for the PiBot Isaac Lab navigation policy",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "policy_controller = rl_controller.controller_node:main",
        ],
    },
)
