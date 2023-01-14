from setuptools import setup,find_packages

setup(
    name='ALDA2022_EmployeeAttrition_P14',
    version='1.0',
    description='An ML based project to predict employee attrition based on relevant attributes and to provide recommendations on ways to reduce attrition rate',
    author='Saksham Pandey, Boscosylvester Chittilapilly, Shlok Naik',
    scripts=['src/main.py'],
    packages=find_packages()
)   
