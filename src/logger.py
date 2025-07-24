# any execution that happens of our code, this file helps us track it.
# it keeps track of all the exceptions, even the custom created ones.

import logging
import os
from datetime import datetime

# creating a log file (text file) (below is the name it will be given)
LOG_FILE = f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"

# providing the path for the log files to get saved in 
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE) # in the current working directory a logs folder will be created and in that a new file of above name will be created and saved
os.makedirs(logs_path, exist_ok=True) # even if the folder exists and there is a file in it, keep appending new files into it 

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)


# format is the format in which the logging message is printed
# level somewhat specifies when the logging will be initiated
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)


if __name__=="__main__":
    logging.info("Logging has started.")