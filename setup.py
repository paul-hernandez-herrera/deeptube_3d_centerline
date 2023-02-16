from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='deeptube_3dcenterline',  
    version = '0.0.1',
    description='Deep learning approach to segment 3D bright-tubular structures, and algorithm to extract the 3D center-line coordinates of the binary tubular image',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='',
    author='Paúl Hernández-Herrera',
    author_email='paul.hernandez@ibt.unam.mx',
    license='BSD 3-clause',
    packages=setuptools.find_packages(),
    python_requires='>=3.7',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: OS Independent',        
        'Programming Language :: Python :: 3.8',
    ],
)