import os
import sys
from pathlib import Path

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
GlOVE_FILE_PATH = os.path.join(PROJECT_ROOT_DIR, 'drivers','glove.6B') 
GLOVE_FILE = 'glove.6B.100d.txt'

def print_msg(message, level=0):
    """ Print the message in formatted way and writes to the log """
    separator = '\t'
    fmt_msg = f'{level * separator}{message}'
    print(fmt_msg)
