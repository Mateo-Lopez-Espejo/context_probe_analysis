from setuptools import find_packages, setup

GENERAL_REQUIRES = ['numpy', 'matplotlib', 'pandas', 'configparser', 'scipy', 'joblib', 'plotly', 'dash']

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='set of tools to calculate and model the effects of past auditory contexts on the response to current '
                ' auditory stimulus',
    author='Mateo Lopez-Espejo',
    install_requires=GENERAL_REQUIRES,
    license='',
)
