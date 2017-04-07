from setuptools import setup

setup(
    name='pybirales',
    version='0.4',
    packages=['pybirales',
              'pybirales.base',
              'pybirales.blobs',
              'pybirales.configuration',
              'pybirales.instrument',
              'pybirales.modules',
              'pybirales.modules.detection',
              'pybirales.modules.monitoring',
              'pybirales.modules.monitoring.api',
              'pybirales.modules.monitoring.api.common',
              'pybirales.modules.monitoring.api.resources',
              'pybirales.plotters'],
    url='https://bitbucket.org/lessju/birales',
    license='',
    author='Alessio Magro',
    author_email='alessio.magro@um.edu.mt',
    description='',
    scripts=['pybirales/scripts/best2_server.py',
             'pybirales/scripts/best2_client.py',
             'pybirales/scripts/best2_process_beams.py'],
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
                      "click"],
)

# TODO: Create /var/log/birales logging directory
