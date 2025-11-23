from setuptools import setup
from stkan import __version__

try:
    with open('README.md', 'r', encoding='utf-8') as fp:
        _long_description = fp.read()
except FileNotFoundError:
    _long_description = ''

setup(
      name='stkan',  # pkg_name
      packages=['stkan',],
      version=__version__,  # version number
      description="Variational autoencoder with Kolmogorov-Arnold Network for spatial domain detection.",
      author='林景',
      author_email='linjing010729@163.com',
      license='MIT',
      url='https://github.com/linjing-lab/stkan',
      download_url='https://github.com/linjing-lab/stkan/tags',
      long_description=_long_description,
      long_description_content_type='text/markdown',
      include_package_data=True,
      zip_safe=False,
      setup_requires=['setuptools>=18.0', 'wheel'],
      project_urls={
            'Source': 'https://github.com/linjing-lab/stkan/tree/main/stkan/',
            'Tracker': 'https://github.com/linjing-lab/stkan/issues',
      },
      classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Intended Audience :: Healthcare Industry',
            'Intended Audience :: Information Technology',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
            'Programming Language :: Python :: 3.12',
            'License :: OSI Approved :: MIT License',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Bio-Informatics',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Software Development',
            'Topic :: Software Development :: Libraries',
            'Topic :: Software Development :: Libraries :: Python Modules',
      ],
      install_requires=[
            'anndata>=0.11.4',
            'faiss-cpu>=1.12.0',
            'matplotlib>=3.10.5',
            'networkx>=3.3',
            'numpy>=1.26.4', # 'numpy==1.26.4'
            'pandas>=2.3.1',
            'pillow>=11.2.1',
            'psutil>=7.0.0',
            'scanpy>=1.10.4',
            'scikit-learn>=1.7.1',
            'scipy>=1.15.3',
            'torch_geometric>=2.3.1',
            'tqdm>=4.67.1',
            'typing_extensions>=4.14.1',
      ],
      # extras_require=[]
)