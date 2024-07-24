from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="beyond-symptoms",
    version="0.1.0",
    author="Jeel Sutariya",
    author_email="sutariyajeel@example.com",
    description="A multi-disease prediction system using machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JeelSutariya/Beyond-Symptoms",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.26.2",
        "pandas>=2.1.4",
        "scikit-learn>=1.3.2",
        "streamlit>=1.29.0",
        "streamlit-option-menu>=0.3.6",
        "matplotlib>=3.8.2",
        "seaborn>=0.13.0",
        "plotly>=5.18.0",
    ],
    entry_points={
        "console_scripts": [
            "beyond-symptoms=app:main",
        ],
    },
)