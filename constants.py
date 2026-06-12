from dotenv import load_dotenv
import os

load_dotenv()

TRAIN = os.getenv('TRAIN')

TRAIN_ANNOTATION = os.getenv('TRAIN_ANNOTATION')

TEST = os.getenv('TEST')

RESULT = os.getenv('RESULT')