import os


descr = """Propensity score matching"""


DISTNAME = 'psmatching'
DESCRIPTION = 'Propensity score matching'
LONG_DESCRIPTION = descr
AUTHOR = 'Chong Dang'
AUTHOR_EMAIL = 'rickydangc@yahoo.com'
URL = 'https://github.com/rickydangc/psmatching'
DOWNLOAD_URL = 'https://github.com/rickydangc/psmatching'
VERSION = '0.0.1'
PYTHON_VERSION = (3.6)


INSTALL_REQUIRES = [
    'numpy',
    'pandas',
    'scipy',
    'statsmodels'
]


if __name__ == "__main__":

    from setuptools import setup
    setup(
        name=DISTNAME,
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        url=URL,
        download_url=DOWNLOAD_URL,

        classifiers=[
            'Development Status :: 3 - Alpha',
            'Environment :: Console',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3',
            'Topic :: Scientific/Engineering',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX',
            'Operating System :: Unix',
            'Operating System :: MacOS',
        ],

        install_requires=INSTALL_REQUIRES,

        packages=['psmatching'],
    )
