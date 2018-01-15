import distutils
import distutils.log as log
import os

from setuptools import setup, find_packages
from setuptools.command.install import install

distutils.log.set_verbosity(distutils.log.info)

HOME = os.environ['HOME']
CONFIG_PATH = 'configuration'
TEMPLATES = os.path.join(CONFIG_PATH, 'templates')


def dir_walk(root_path):
    """
    Iterate over all the files (including sub-directories) of the root path

    :param root_path:
    :return:
    """

    file_paths = []
    for file_path, _, files in os.walk(root_path):
        print(file_path, files)
        for name in files:
            file_paths.append(os.path.join(file_path, name))

    return file_paths


class InstallWrapperCommand(install):
    """
    A wrapper function for the standard setuptools installation command
    in order to add Custom Pre and Post processing commands

    """
    LOCAL_DIRS = [
        '/var/log/birales',
        os.path.join(HOME, '.birales'),
        os.path.join(HOME, '.birales/configuration/templates/dev'),
        os.path.join(HOME, '.birales/configuration/templates/prod'),
        os.path.join(HOME, '.birales/schedules'),
        os.path.join(HOME, '.birales/visualisation/fits'),
    ]

    def run(self):
        # Run the (custom) initialisation functions
        self._init()
        # Run the standard PyPi copy
        install.run(self)

    def _init(self):
        # Pre-processing section
        log.info('Initialising local directories in ~/.birales')
        for d in self.LOCAL_DIRS:
            if not os.path.exists(d):
                log.info('Created directory in {}'.format(d))
                os.makedirs(d)

    def _check(self):
        # todo - add sanity checks here (check that all the dependencies are installed)
        pass


setup(
    name='pybirales',
    version='2.0',
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
    zip_safe=False,
    install_requires=['configparser', 'futures', 'enum34', 'astropy', 'astroplan', 'numpy', 'matplotlib', 'numba',
                      'pymongo==2.8.*', 'scipy', 'scikit-learn', 'ephem', 'h5py', 'click', 'flask', 'flask-compress',
                      'flask-socketio', 'flask_ini', 'fadvise', 'sklearn', 'pandas', 'webargs', 'yappi', 'marshmallow',
                      'humanize', 'mongoengine', 'pyfits', 'construct==2.5.5', 'corr', 'python-dateutil',
                      'slackclient'],
    data_files=[
        (os.path.join(HOME, '.birales', CONFIG_PATH), dir_walk(os.path.join('pybirales', CONFIG_PATH))),
        (os.path.join(HOME, '.birales', TEMPLATES, 'dev'), dir_walk(os.path.join('pybirales', TEMPLATES, 'dev'))),
        (os.path.join(HOME, '.birales', TEMPLATES, 'prod'), dir_walk(os.path.join('pybirales', TEMPLATES, 'prod'))),
        (os.path.join(HOME, '.birales/fits'), []),
        ('/var/log/birales', [])
    ],
    entry_points={
        'console_scripts': [
            'birales = pybirales.cli.cli:cli',
            'birales-frontend = pybirales.app.app:main'
        ]
    },
    cmdclass={
        'install': InstallWrapperCommand,  # override the standard installation command to add custom logic
    },
)
