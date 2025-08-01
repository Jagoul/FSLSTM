from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="fslstm",
    version="1.0.0",
    author="FSLSTM Contributors",
    description="A privacy-by-design federated learning framework for anomaly detection in smart buildings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/FSLSTM",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.910",
            "pre-commit>=2.15",
        ],
        "visualization": [
            "plotly>=5.0",
            "dash>=2.0",
            "streamlit>=1.0",
        ],
        "gpu": [
            "torch>=1.7.0+cu110",
            "torchvision>=0.8.0+cu110",
        ]
    },
    entry_points={
        "console_scripts": [
            "fslstm-train=scripts.train:main",
            "fslstm-evaluate=scripts.evaluate:main",
            "fslstm-preprocess=scripts.preprocess_data:main",
        ],
    },
    include_package_data=True,
    package_data={
        "fslstm": ["configs/*.yaml", "data/examples/*.csv"],
    },
    zip_safe=False,
)