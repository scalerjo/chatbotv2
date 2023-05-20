import glob
import fitz
import docx
import json
import jsonlines
from tqdm import tqdm
import time
import validators
from bs4 import BeautifulSoup
import requests
import openai
import numpy as np
from numpy.linalg import norm
import os
import hashlib
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter, PythonCodeTextSplitter, MarkdownTextSplitter
from constants import MarkdownExtensions, PythonExtensions, GolangExtensions
from custom_text_splitter import GolangCodeTextSplitter, BaseTextSplitter
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

tokenizer = tiktoken.get_encoding("cl100k_base")

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_embedding(text, model="text-embedding-ada-002"):
	return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def chatGPT_api(messages):
	completion = openai.ChatCompletion.create(
	model = 'gpt-3.5-turbo',
	messages=messages,
	temperature = 1,
	top_p = 0.95,
	# max_tokens=2000,
	frequency_penalty = 0.0,
	presence_penalty = 0.0
	)

	return completion.choices[0].message

def get_summary(chunk):
	content = "The following is a passage fragment. Please summarize what information the readers can take away from it:"
	content += "\n" + chunk
	messages = [
				{"role": "user", "content": content}
			]
	summary = chatGPT_api(messages).content
	return summary

def split_into_chunks(dir):
	chunks = []
	paths = glob.glob(dir, recursive = True)

	text_splitter_options = {
		"chunk_size": 800,
		"chunk_overlap": 0
	}
	base_splitter = BaseTextSplitter(**text_splitter_options)
	python_splitter = PythonCodeTextSplitter(**text_splitter_options)
	markdown_splitter = MarkdownTextSplitter(**text_splitter_options)
	golang_splitter = GolangCodeTextSplitter(**text_splitter_options)

	report = {}

	for path in paths:
		with open(path) as f:
			file = f.read()
			f.close()

			if any([path.endswith(ext) for ext in PythonExtensions]):
				texts = python_splitter.create_documents([file])
			elif any([path.endswith(ext) for ext in MarkdownExtensions]):
				texts = markdown_splitter.create_documents([file])
			elif any([path.endswith(ext) for ext in GolangExtensions]):
				texts = golang_splitter.create_documents([file])
			else:
				texts = base_splitter.create_documents([file])
			chunks.extend([text.page_content for text in texts])


	return (chunks, report)

def store_info(dir, memory_path, chunk_sz = 800, max_memory = 100):
	info = []

	(chunks, _) = split_into_chunks(dir)

	for chunk in tqdm(chunks):
		summary = get_summary(chunk)
		embd = get_embedding(chunk)
		summary_embd = get_embedding(summary)
		item = {
			"id": len(info),
			"text": chunk,
			"embd": embd,
			"summary": summary,
			"summary_embd": summary_embd,
		}
		info.append(item)
		time.sleep(3)  # up to 20 api calls per min
	
	# Store brain to memory path
	with jsonlines.open(memory_path, mode="w") as f:
		f.write(info)
		print("Finish storing info.")

def memorize(dir, memory_path):
	print("Memorizing...")
	store_info(dir, memory_path)
	return memory_path







class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def load_info(memory_path):
	with open(memory_path, 'r', encoding='utf8') as f:
		for line in f:
			info = json.loads(line)
	return info

def retrieve(q_embd, info):
	# return the indices of top three related texts
	text_embds = []
	summary_embds = []
	for item in info:
		text_embds.append(item["embd"])
		summary_embds.append(item["summary_embd"])
	# compute the cos sim between info_embds and q_embd
	text_cos_sims = np.dot(text_embds, q_embd) / (norm(text_embds, axis=1) * norm(q_embd))
	summary_cos_sims = np.dot(summary_embds, q_embd) / (norm(summary_embds, axis=1) * norm(q_embd))
	cos_sims = text_cos_sims + summary_cos_sims
	top_args = np.argsort(cos_sims).tolist()
	top_args.reverse()
	indices = top_args[0:3]
	return indices

def get_qa_content(q, retrieved_text):
	content = "After reading some relevant passage fragments from the same document, please respond to the following query. Note that there may be typographical errors in the passages due to the text being fetched from a PDF file or web page."

	content += "\nQuery: " + q

	for i in range(len(retrieved_text)):
		content += "\nPassage " + str(i + 1) + ": " + retrieved_text[i]

	content += "\nAvoid explicitly using terms such as 'passage 1, 2 or 3' in your answer as the questioner may not know how the fragments are retrieved. You can use your own knowledge in addition to the provided information to enhance your response. Please use the same language as in the query to respond, to ensure that the questioner can understand."

	return content

def generate_answer(q, retrieved_indices, info):
	while True:
		sorted_indices = sorted(retrieved_indices)
		retrieved_text = [info[idx]["text"] for idx in sorted_indices]
		content = get_qa_content(q, retrieved_text)
		if len(tokenizer.encode(content)) > 3800:
			retrieved_indices = retrieved_indices[:-1]
			print("Contemplating...")
			if not retrieved_indices:
				raise ValueError("Failed to respond.")
		else:
			break
	messages = [
		{"role": "user", "content": content}
	]
	answer = chatGPT_api(messages).content
	return answer

def answer(q, info):
	q_embd = get_embedding(q, model="text-embedding-ada-002")
	retrieved_indices = retrieve(q_embd, info)
	answer = generate_answer(q, retrieved_indices, info)
	return answer

def chat(memory_path):
	info = load_info(memory_path)
	while True:
		q = input("Enter your question: ")
		if len(tokenizer.encode(q)) > 200:
			raise ValueError("Input query is too long!")
		response = answer(q, info)
		print()
		print(f"{bcolors.OKGREEN}{response}{bcolors.ENDC}")
		print()
		time.sleep(3)