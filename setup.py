from setuptools import setup, find_packages

setup(name='mediator_was',
      version='0.1',
      description='Mediator-wide association study toolbox',
      url='http://github.com/Schork-Lab/mediator-was',
      author='Kunal Bhutani, Abhishek Sarkar',
      author_email='kbhutani@ucsd.edu',
      license='MIT',
      packages=find_packages(),
      entry_points={
          'console_scripts': [
              'mediator-was-wtccc=mediator_was.processing.wtccc:fit',
              'mediator-was-twas=mediator_was.twas.bare'
          ]
      },
      zip_safe=False)
