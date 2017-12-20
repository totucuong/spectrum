from setuptools import setup

setup(name='spectrum',
      version='0.1',
      description='A python library for data fusion',
      url='https://github.com/totucuong/spectrum',
      author='To Tu Cuong',
      author_email='to.cuong@fu-berlin.de',
      license='Apache-2.0',
      packages=['spectrum', 'spectrum.datasets', 'spectrum.inferences', 'spectrum.models'],
      extras_require={
          'numpy': ['numpy>=1.13.3']
      },
      zip_safe=False)