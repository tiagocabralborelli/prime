from setuptools import setup, find_packages

setup(
    name="prime",
    version="1.0.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    py_modules=[
        "Clean",
        "UncertTorch",
        "Inference",
    ],
    package_data={
        "": ["*.yaml"],
    },
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "prime-process = Clean:main",
            "prime-uncert = UncertTorch:main",
            "prime-predict = Inference:main",
        ]
    },
)
