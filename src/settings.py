import os
from dotenv import load_dotenv

load_dotenv()

PRODUCTION_MODEL_FOLDER = os.getenv("PRODUCTION_MODEL_FOLDER")
