from setuptools import setup, find_packages

setup(
    name='pybirales',
    version='1.0',
    packages=find_packages(),
    url='https://bitbucket.org/lessju/birales',
    license='',
    author='Alessio Magro',
    author_email='alessio.magro@um.edu.mt',
    description='',
    scripts=['pybirales/services/scripts/best2_server.py',
             'pybirales/services/scripts/best2_client.py',
             'pybirales/services/scripts/best2_process_beams.py',
             'pybirales/cli/cli.py'],
    install_requires=['configparser',
                      'futures',
                      'enum34',
                      "astropy",
                      "astroplan",
                      "numpy",
                      "matplotlib",
                      "numba",
                      "pymongo",
                      "scipy",
                      "scikit-learn",
                    #  "h5py",
                      "click",
                      "flask",
                      "flask-compress",
                      'sklearn', 'pandas', 'bson', 'webargs', 'yappi'],
    entry_points={
        'console_scripts': [
            'birales = pybirales.cli.cli:cli',
            'birales-frontend = pybirales.app.api:run_server'
        ]
    },
)

# TODO: Create /var/log/birales logging directory
