from setuptools import setup, find_packages

setup(
    name="esm-prime",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    entry_points={
        "console_scripts": [
            "prime-train =  Train:main",
            "prime-predict = Predict:main",
            "prime-uncertainty = Uncert:main",
            "prime-torchuncert = UncertTorch:main",
            "prime-process = Clean:main", 
        ]
    }
)
