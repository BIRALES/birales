import distutils.log as log
import os

from setuptools import setup, find_packages
from setuptools.command.install import install

HOME = os.environ['HOME']
CONFIG_PATH = 'configuration'
TEMPLATES = os.path.join(CONFIG_PATH, 'templates')
DEPENDENCIES = []


def list_dir(root_path):
    """
    Iterate over all the files (including sub-directories) of the root path

    :param root_path:
    :return:
    """

    file_paths = []
    for item in os.listdir(root_path):
        file_path = os.path.join(root_path, item)
        if os.path.isfile(file_path):
            file_paths.append(file_path)
    return file_paths


class InstallWrapperCommand(install):
    """
    A wrapper function for the standard setuptools installation command
    in order to add Custom Pre- and Post-processing commands

    """
    LOCAL_DIRS = [
        #  '/var/log/birales',
        os.path.join(HOME, '.birales'),
        os.path.join(HOME, '.birales/configuration/templates/dev'),
        os.path.join(HOME, '.birales/schedules'),
        os.path.join(HOME, '.birales/visualisation/fits'),
        os.path.join(HOME, '.birales/tdm/out'),
        os.path.join(HOME, '.birales/tdm/in')
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
    version='3.1',
    packages=find_packages(),
    url='https://bitbucket.org/lessju/birales',
    license='',
    author='Alessio Magro',
    author_email='alessio.magro@um.edu.mt',
    description='',
    scripts=['pybirales/cli/cli.py',
             'frontend/app.py'],
    include_package_data=True,
    zip_safe=False,
    install_requires=DEPENDENCIES,
    setup_requires=DEPENDENCIES,
    data_files=[
        (os.path.join(HOME, '.birales', TEMPLATES), []),
        (os.path.join(HOME, '.birales/fits'), []),
        (os.path.join(HOME, '.birales/configuration.py/uploaded'), []),
    ],
    entry_points={
        'console_scripts': [
            'birales = pybirales.cli.cli:cli',
            'birales-frontend = frontend.app:main'
        ]
    },
    cmdclass={
        'install': InstallWrapperCommand,  # override the standard installation command to add custom logic
    },
)
