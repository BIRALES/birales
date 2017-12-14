import os
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
             'pybirales/cli/cli.py',
             'pybirales/app/app.py'
             ],
    include_package_data=True,
    #package_data={'pybirales/app/templates': ['*'], 'pybirales/app/static': ['*']},
    zip_safe=False,
    install_requires=['configparser',
                      'futures',
                      'enum34',
                      "astropy",
                      "astroplan",
                      "numpy",
                      "matplotlib",
                      "numba",
                      "pymongo==2.8.*",
                      "scipy",
                      "scikit-learn",
                      #  "h5py",
                      "click",
                      "flask",
                      "flask-compress",
                      "flask-socketio",
                      "flask_ini",
                      "fadvise",
                      'sklearn', 'pandas', 'webargs', 'yappi', 'marshmallow', 'humanize', 'mongoengine', 'pyfits',
                      'construct==2.5.5-reupload',
                      'corr'],
    data_files=[(os.environ['HOME'] + '/.birales', ['pybirales/configuration/local.ini']),
                (os.environ['HOME'] + '/.birales/tcpo', []),
                (os.environ['HOME'] + '/.birales/fits', []),
                ('/var/log/birales', [])
                ],
    entry_points={
        'console_scripts': [
            'birales = pybirales.cli.cli:cli',
            'birales-frontend = pybirales.app.app:main'
        ]
    },
)

# TODO: Create /var/log/birales logging directory
