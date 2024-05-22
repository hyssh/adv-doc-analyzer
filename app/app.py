import os
import sys
import chainlit as cl
from dotenv import load_dotenv
from utils import get_env

sys.path.append(os.getcwd())

env_file_path = '.env'
load_dotenv(env_file_path, override=True)
get_env('OPENAI_API_BASE')
