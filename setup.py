from setuptools import setup, find_packages
from typing import List

def get_requirements(file_to_path:str)-> List[str]:
    """This function creates the list of requirements

    Args:
        file_to_path (str): library names

    Returns:
        List[str]: list of the library names
    """
    requirements = []
    HYPHEN_E_DOT = "-e ."
    
    with open(file_to_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)    
    return requirements
        
        
setup(
    name = "mlproject",
    version= "0.0.1",
    author= "sayed_delwoar",
    author_email= "sahed.delowar@gmail.com",
    packages= find_packages(),
    install_requires= get_requirements('requirements.txt')
    
)