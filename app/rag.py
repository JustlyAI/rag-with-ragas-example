import os
import tiktoken
import json
import requests
import asyncio
from openai import AsyncOpenAI
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from termcolor import colored
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()


class Rag:
    def __init__(self):
        self.token_encoding_model = "gpt-4"
        self.embedding_model = "text-embedding-3-large"
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", "YOUR_API_KEY"))

    async def fetch_text_from_url(self, url):
        """Fetches text from a specified URL."""
        response = requests.get(
            f"https://r.jina.ai/{url}", headers={"X-Return-Format": "text"}
        )
        response.raise_for_status()
        return response.text

    def load_text_file(self, file_path):
        """Loads text from a file located at the specified path."""
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    def load_pdf_file(self, file_path):
        """Loads text from a PDF file located at the specified path."""
        with open(file_path, "rb") as file:
            reader = PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
            return text

    def count_tokens(self, text):
        """Counts the number of tokens in the given text using the specified token encoding model."""
        enc = tiktoken.encoding_for_model(self.token_encoding_model)
        return len(enc.encode(text))

    def process_text(self, text):
        """Processes text into chunks suitable for embedding."""
        enc = tiktoken.encoding_for_model(self.token_encoding_model)
        encoded = enc.encode(text)
        chunks = []
        chunk_size = 800
        overlap = 400
        for i in range(0, len(encoded), chunk_size - overlap):
            chunk = encoded[i : i + chunk_size]
            chunks.append(chunk)
        decoded_chunks = [enc.decode(chunk) for chunk in chunks]
        return decoded_chunks

    async def embed_text_chunks(self, chunks):
        """Embeds text chunks asynchronously."""

        async def embed_chunk(chunk):
            response = await self.client.embeddings.create(
                input=chunk, model=self.embedding_model
            )
            return {"text": chunk, "embedding": response.data[0].embedding}

        tasks = [embed_chunk(chunk) for chunk in chunks]
        embedded_chunks = await asyncio.gather(*tasks)
        return embedded_chunks

    async def embed_text_chunks_for_eval(self, chunks):
        """Embeds text chunks for evaluation purposes asynchronously."""

        async def embed_chunk_for_eval(chunk):
            response = await self.client.embeddings.create(
                input=chunk, model=self.embedding_model
            )
            return Document(
                page_content=chunk, metadata={"embedding": response.data[0].embedding}
            )

        tasks = [embed_chunk_for_eval(chunk) for chunk in chunks]
        embedded_chunks = await asyncio.gather(*tasks)
        return embedded_chunks

    def save_chunks_to_file(self, chunks, filename="app/output/chunks.json"):
        """Saves text chunks to a file."""
        with open(filename, "w", encoding="utf-8") as json_file:
            json.dump(chunks, json_file, indent=4)

    def save_chunks_to_file_for_eval(
        self, chunks, filename="app/output/data_chunks.json"
    ):
        """Saves text chunks used for evaluation to a file."""
        chunks_dict = [
            {"page_content": chunk.page_content, "metadata": chunk.metadata}
            for chunk in chunks
        ]
        with open(filename, "w", encoding="utf-8") as json_file:
            json.dump(chunks_dict, json_file, indent=4)

    async def process_files_in_folder_for_eval(self, data_folder_path):
        """Processes all files in a specified folder for evaluation."""
        if os.path.exists("app/output/chunks.json"):
            with open(
                "app/output/data_chunks.json", "r", encoding="utf-8"
            ) as json_file:
                all_chunks = json.load(json_file)
                if isinstance(all_chunks[0], dict):
                    all_chunks = [
                        Document(
                            page_content=chunk["page_content"],
                            metadata=chunk["metadata"],
                        )
                        for chunk in all_chunks
                    ]
                else:
                    all_chunks = [Document(page_content=chunk) for chunk in all_chunks]
        else:
            all_chunks = []
            for filename in os.listdir(data_folder_path):
                file_path = os.path.join(data_folder_path, filename)
                if filename.endswith(".txt"):
                    text = self.load_text_file(file_path)
                elif filename.endswith(".pdf"):
                    text = self.load_pdf_file(file_path)
                else:
                    continue
                chunks = self.process_text(text)
                for chunk in chunks:
                    document = Document(
                        page_content=chunk, metadata={"file_name": filename}
                    )
                    all_chunks.append(document)
                print(colored(f"Processed {filename}", "green"))

        # Embed the text chunks
        all_chunks = await self.embed_text_chunks_for_eval(
            [chunk.page_content for chunk in all_chunks]
        )

        # Save all chunks to a file
        self.save_chunks_to_file_for_eval(
            all_chunks, filename="app/output/data_chunks.json"
        )

        return all_chunks

    def clear_output_folder(self):
        """Clears all JSON and CSV files in the 'app/output' directory."""
        for filename in os.listdir("app/output"):
            if filename.endswith(".json") or filename.endswith(".csv"):
                file_path = os.path.join(
                    "app/output", filename
                )  # Construct the full file path
                os.remove(file_path)  # Remove the file at the given path

    async def embed_query(self, query, prequery="", postquery=""):
        """Embeds a query with optional pre and post text."""
        full_query = f"{prequery} {query} {postquery}"
        response = await self.client.embeddings.create(
            input=full_query, model=self.embedding_model
        )
        return response.data[0].embedding

    def cosine_similarity_search(self, query_embedding, embedded_chunks, top_k=5):
        """Performs a cosine similarity search between a query embedding and a list of text embeddings."""
        chunk_embeddings = np.array([chunk["embedding"] for chunk in embedded_chunks])
        similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
        top_k_indices = similarities.argsort()[-top_k:][::-1]
        return [embedded_chunks[i]["text"] for i in top_k_indices]

    def save_top_chunks_text_to_file(
        self, top_chunks, filename="app/output/top_chunks.json"
    ):
        """Saves the top chunks of text to a file."""
        top_chunks_text = [chunk for chunk in top_chunks]
        with open(filename, "w", encoding="utf-8") as json_file:
            json.dump(top_chunks_text, json_file, indent=4)

    async def call_gpt4(self, messages):
        """Calls the GPT-4 model to generate responses based on the provided messages."""
        response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            stream=False,
        )
        assistant_response = response.choices[0].message.content
        print(colored(assistant_response, "green"))
        return assistant_response

    async def call_gpt4_with_streaming(self, messages):
        """Calls the GPT-4 model with streaming enabled to generate responses based on the provided messages."""
        response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            stream=True,
        )
        assistant_response = ""
        async for chunk in response:
            if chunk.choices[0].delta.content is not None:
                assistant_response += chunk.choices[0].delta.content
                print(
                    colored(chunk.choices[0].delta.content, "green"), end="", flush=True
                )
        return assistant_response

    async def call_gpt4_with_streaming_for_streamlit(self, messages):
        """Calls the GPT-4 model with streaming enabled to generate responses based on the provided messages."""
        response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            stream=True,
        )
        async for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
                print(
                    colored(chunk.choices[0].delta.content, "green"), end="", flush=True
                )

    async def call_gpt4_with_json(self, messages):
        """Calls the GPT-4 model to generate JSON formatted responses based on the provided messages."""
        messages.insert(
            0,
            {
                "role": "system",
                "content": "You are a helpful assistant. Please provide the response in JSON format.",
            },
        )
        response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            response_format={"type": "json_object"},
            stream=False,
        )
        assistant_response = response.choices[0].message.content
        print(colored(assistant_response, "green"))
        return assistant_response
