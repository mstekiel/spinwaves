from setuptools import setup

setup(name='spinwaves',
    version='0.1.0',
    description='Linear spin wave theory in python as in SpinW',
    # url='https://github.com/mstekiel/mikibox',
    author='Michal Stekiel',
    author_email='michal.stekiel@gmail.com',
    packages=['spinwaves'],
    python_requires='==3.9.*',
    install_requires=[
        'numpy==1.23.5',
        'matplotlib==3.5.2',
        'scipy==1.9.1',
        'vispy==0.12.1'
    ],
    extras_require={
        "dev": ['sphinx==5.0.2', 'pandas==1.4.4'],
    },
    include_package_data=True,
    zip_safe=False)