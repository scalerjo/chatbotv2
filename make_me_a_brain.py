from dotenv import load_dotenv
import openai
import os
from utils import memorize

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

if __name__ == "__main__":
    memory_path = "./embeddings/brain.json"
    memorize('./data/**/*', memory_path)