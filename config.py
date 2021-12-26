import os
import sys
from pathlib import Path

#if Path(__file__).resolve().parent not in sys.path:
#    sys.path.append(Path(__file__).resolve().parent.parent.parent)

#PROJECT_ROOT_DIR = Path(os.getcwd()).resolve().parent
PROJECT_ROOT_DIR = Path(__file__).resolve().parent
#Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(PROJECT_ROOT_DIR, 'data')
DATA_FILES_DIR = os.path.join(DATA_DIR, 'dump')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'to_annotate')
ANNOTATED_DIR = os.path.join(DATA_DIR, 'annotated')
CLUSTERED_DIR = os.path.join(DATA_DIR, 'clustered')
WEBDRIVER_LOC = os.path.join(PROJECT_ROOT_DIR, 'drivers','chromedriver_win32','chromedriver.exe')
BASE_URL = 'http://www.instagram.com'
CSV_DIR = os.path.join(PROJECT_ROOT_DIR, 'results')
GAPP_CRED = os.path.join(PROJECT_ROOT_DIR, 'drivers','instaphotolabelling-e33933c7b993.json')
EXCLUDE = ['sky','font','rectangle']
TOURISM_ANCHORS = ['sunset','places','art','beach','snow','mountain','shopping','food','history','fun','transport','water','chirstmas','happy']
W2V_MODEL_FPATH = os.path.join(PROJECT_ROOT_DIR, 'core','model')
W2V_MODEL_FNAME = os.path.join(W2V_MODEL_FPATH, 'W2VTourismModel.model')
GlOVE_FILE_PATH = os.path.join(PROJECT_ROOT_DIR, 'drivers','glove.6B') 
GLOVE_FILE = 'glove.6B.100d.txt'
MAPQKEY = 'cTSIg6TdVjgOx6hNUPgqs20QtWQwOljU'
MAPQURL='http://www.mapquestapi.com/geocoding/v1/address'

def print_msg(message, level=0):
    """ Print the message in formatted way and writes to the log """
    separator = '\t'
    fmt_msg = f'{level * separator}{message}'
    print(fmt_msg)