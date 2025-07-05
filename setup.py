# C:\Tenx-projects\credit-risk-model\setup.py
from setuptools import setup, find_packages

setup(
    name='credit_risk_model',
    version='0.1.0',
    packages=find_packages(), # This line is critical for finding your 'src' package
    install_requires=[
        # List your project's dependencies here, e.g.:
        # 'pandas>=1.0.0',
        # 'scikit-learn>=1.0.0',
        # 'numpy>=1.20.0',
        # 'fastapi',
        # 'uvicorn',
        # 'pydantic',
    ],
    author='Your Name',
    author_email='your.email@example.com',
    description='A credit risk prediction model',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/credit-risk-model',
)