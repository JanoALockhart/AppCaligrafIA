import os
from dotenv import load_dotenv

load_dotenv()

PRODUCTION_MODEL_FOLDER = os.getenv("PRODUCTION_MODEL_FOLDER")
TEST_IMG_FILE = os.getenv("TEST_IMG_FILE")