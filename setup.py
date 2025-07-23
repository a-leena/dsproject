from setuptools import find_packages,setup
HYPHEN_E_DOT = '-e .'

def get_requirements(file_path):
    with open(file_path) as file_obj:
        requirements = [req.replace("\n","") for req in file_obj.readlines()]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
        return requirements

# metadata about the entire project
setup(
    name='dsproject',
    version='0.0.1',
    author='Aleena',
    packages=find_packages(), # init file inside src helps in finding the packages
    install_requires=get_requirements('requirements.txt')
)
