from setuptools import setup, find_packages

setup(
    name="cline",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow",
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn"
    ],
    entry_points={
        'console_scripts': [
            'cline-main=main:main',
            'cline-test=test_install:main',
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A simple project demonstrating the use of common data science and machine learning libraries.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/yourusername/cline",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
