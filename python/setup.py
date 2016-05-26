from setuptools import setup

setup(
    name='pybirales',
    version='0.4',
    packages=['pybirales', 'pybirales.base', 'pybirales.blobs',
              'pybirales.configuration', 'pybirales.instrument',
              'pybirales.modules', 'pybirales.plotters'],
    url='https://bitbucket.org/lessju/birales',
    license='',
    author='Alessio Magro',
    author_email='alessio.magro@um.edu.mt',
    description='',
    scripts=['pybirales/scripts/best2_server.py', 'pybirales/scripts/best2_client.py'],
    install_requires=['futures', 'enum34', "astropy", "astroplan",
                      "numpy", "matplotlib", "numba"],
)

# TODO: Creatre /var/log/birales logging directory
