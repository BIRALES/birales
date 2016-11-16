from distutils.core import setup

setup(
    name='pybirales-post',
    version='1.0',
    packages=['', 'core', 'configuration', 'visualization', 'visualization.api', 'visualization.api.common',
              'visualization.api.resources'],
    package_dir={'': 'postprocessing'},
    url='https://bitbucket.org/lessju/birales',
    license='',
    author='Denis',
    author_email='denis.cutajar@um.edu.mt',
    description='BIRALES post processing module'
)
