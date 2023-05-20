import os
from dotenv import load_dotenv
import openai
from utils import chat
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

chat("./embeddings/brain.json")

