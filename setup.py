from setuptools import setup
setup(
    name = 'balancer',
    version = '0.1.0',
    packages = ['balancer'],
    entry_points = {
        'console_scripts': [
            'balancer = balancer.__main__:main'
        ]
    })