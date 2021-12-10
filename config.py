import os
import sys
from pathlib import Path

#if Path(__file__).resolve().parent not in sys.path:
#    sys.path.append(Path(__file__).resolve().parent.parent.parent)

PROJECT_ROOT_DIR = Path(os.getcwd()).resolve().parent
#Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(PROJECT_ROOT_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'to_annotate')
ANNOTATED_DIR = os.path.join(DATA_DIR, 'annotated')
CLUSTERED_DIR = os.path.join(DATA_DIR, 'clustered')
IMAGE_DIR = os.path.join(PROJECT_ROOT_DIR, 'image')
WEBDRIVER_LOC = 'C:/Docs/chromedriver_win32/chromedriver.exe'
BASE_URL = 'http://www.instagram.com'
CSV_DIR = os.path.join(PROJECT_ROOT_DIR, 'results')
GAPP_CRED = 'C:/Users/nkushwaha/instaphotolabelling-e33933c7b993.json'
GlOVE_FILE_PATH = 'C:/Docs/uslm/reference/glove.6B/'
GLOVE_FILE = 'glove.6B.100d.txt'

def print_msg(message, level=0):
    """ Print the message in formatted way and writes to the log """
    separator = '\t'
    fmt_msg = f'{level * separator}{message}'
    print(fmt_msg)