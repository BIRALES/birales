from setuptools import setup, find_packages

setup(
    name='pybirales',
    version='0.4',
    packages=find_packages(),
    url='https://bitbucket.org/lessju/birales',
    license='',
    author='Alessio Magro',
    author_email='alessio.magro@um.edu.mt',
    description='',
    scripts=['pybirales/back-end/services/scripts/best2_server.py',
             'pybirales/back-end/services/scripts/best2_client.py',
             'pybirales/back-end/services/scripts/best2_process_beams.py',
             'pybirales/back-end/birales.py'],
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
                      "h5py",
                      "click",
                      "flask",
                      "flask-compress",
                      'sklearn', 'pandas', 'bson', 'webargs'],
)

# TODO: Create /var/log/birales logging directory
