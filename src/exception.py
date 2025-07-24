# this will be common for the entire code whenever try catch will be used anywhere in our program.

# look into the exception documentation for python online for some standard codes

import sys
from src.logger import logging

# def error_message_detail(error, error_detail:sys): -> the ":sys" syntax is called Type annotation or type hinting, 
# we're kind of indicating what datatype of value that parameter is expected to have; 
# it's not necessary to be used;
# but it helps in function definition as we can see the definitions and required arguments of other methods 
# or the other associated variables to that variable may have via its said type.

def error_message_detail(error, error_detail:sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occured in Python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message

# Exception in parentheses means we're inheriting from Exception class
class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message) # inheriting the init function of Exception class
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message
    


if __name__== "__main__":
    try:
       a=1/0
    except Exception as e:
        logging.info("Divide by 0 error for testing exception file.")
        raise CustomException(e, sys)