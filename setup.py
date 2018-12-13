from setuptools import setup, find_packages


try:  # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError:  # for pip <= 9.0.3
    from pip.req import parse_requirements


# parse_requirements() returns generator of pip.req.InstallRequirement objects
install_reqs = parse_requirements('requirements.txt', session='null')


# reqs is a list of requirement
# e.g. ['django==1.5.1', 'mezzanine==1.4.6']
reqs = [str(ir.req) for ir in install_reqs]


with open('README.md') as f:
    long_description = f.read()


setup(name='gluon2pytorch',
      version='0.0.3',
      description='The deep learning models convertor',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/nerox8664/gluon2pytorch',
      author='Grigory Malivenko',
      author_email='nerox8664@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=reqs,
      zip_safe=False)
