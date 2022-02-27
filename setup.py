# -*- coding: utf-8 -*-

from os import path
import setuptools

readme_file = path.join(path.dirname(path.abspath(__file__)), 'README.md')
try:
    from m2r import parse_from_file
    readme = parse_from_file(readme_file)
except ImportError:
    with open(readme_file) as f:
        readme = f.read()

# install_requires = ['xml']
install_requires=[
    'numpy', 
    'dpdata',
    # 'dflow',
]

setuptools.setup(
    name="dpgen2",
    use_scm_version={'write_to': 'dpgen2/_version.py'},
    setup_requires=['setuptools_scm'],
    author="Han Wang",
    author_email="wang_han@iapcm.ac.cn",
    description="Deep potential generator, version 2",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/deepmodeling/dpgen2",
    packages=['dpgen2',
              'dpgen2/op',
              'dpgen2/superop',
              'dpgen2/utils',
              'dpgen2/fp',
    ],
    package_data={'dpgen2':['*.json']},
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
    ],
    keywords='deep potential concurrent learning',
    install_requires=install_requires,
    extras_require={
        'docs': ['sphinx', 'recommonmark', 'sphinx_rtd_theme>=1.0.0rc1', 'numpydoc', 'm2r2'],
    }
)

