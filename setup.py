import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'multi_camera_launch'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        # CRITICAL: Install asset directories into the workspace share path
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'calibration'), glob('calibration/*.yaml')),
    ],
    install_requires=['setuptools', 'pyyaml'],
    zip_safe=True,
    maintainer='developer',
    maintainer_email='you@domain.com',
    description='High-performance zero-copy multi-camera perception stack pipeline',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)
