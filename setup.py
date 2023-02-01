from setuptools import setup
setup(
    name = 'proxmox-load-balancer',
    version = '0.1.0',
    packages = ['proxmox-load-balancer'],
    entry_points = {
        'console_scripts': [
            'proxmox-load-balancer = proxmox-load-balancer.__main__:main'
        ]
    })