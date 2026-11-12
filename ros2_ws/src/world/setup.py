from setuptools import setup

package_name = "world"

setup(
    name=package_name,
    version="1.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Stav Mordekhai",
    description="Simulated world: track, cones, car physics and cone sensor. Publishes sensor data, receives control commands.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "world_node = world.world_node:main",
        ],
    },
)
