# Basic RAG 📚 with RAGAS Evaluation ✅

#### Build and evaluate your own RAG system in (almost) plain python. This is a fully functional and highly modeularized RAG system that can be used as a base for building custom RAG systems.

### Prerequisites

- IDE (VSCode, PyCharm, Jupyter Notebook, etc.)
- Python 3.10 or later
- Anaconda (recommended)
- Docker (optional)
- OpenAI API Key

### Setting Up Environment Variables

Before running the application, ensure you have set the `OPENAI_API_KEY` in a `.env` file in the root directory of your project. This key is necessary for accessing OpenAI's API. To obtain key, visit [Overview - OpenAI API](https://platform.openai.com/docs/overview).

1. **Create a `.env` file in the root directory:**

   ```sh
   touch .env
   ```

2. **Add your OpenAI API key to the `.env` file:**

   ```sh
   echo "OPENAI_API_KEY=your_openai_api_key_here" >> .env
   ```

Replace `your_openai_api_key_here` with your actual OpenAI API key.

The current implementation defaults to gpt-4-turbo for the LLM model and OpenAI's text-embedding-3-large for the embeddings model.

## Quick Start

**Run these 3 commands to check the application components:**(May not work on all systems, in any case see Docker Instructions)

```sh
python -m app.chat
python -m app.eval
streamlit run app/streamlit.py
```

A deployed basic RAG with RAGAS Streamlit App can be accessed directly at http:// (bring your own key).

### Docker Instructions

#### Build and Run Docker Image

1. **Build the Docker Image Locally:**

   For Windows:

   ```sh
   docker build -t rag-ragas-test .
   ```

   For Mac Silicon (M1, M2, M3, M4):

   ```sh
   docker buildx build --platform linux/amd64 -t rag-ragas-test . --load
   ```

2. **Verify the Image Exists Locally:**

   ```sh
   docker images
   ```

3. **Run the Docker Container:**

   ```sh
   docker run -p 8080:8080 rag-ragas-test
   ```

### Acknowledgements

This project has been inspired and built upon the work of brilliant and generous individuals, namely:

- [Modular Rag and chat implementation from URLs, PDFs and txt files. | Patreon](https://www.patreon.com/posts/modular-rag-and-106461497)
- [Coding-Crashkurse/RAG-Evaluation-with-Ragas](https://github.com/Coding-Crashkurse/RAG-Evaluation-with-Ragas)

### Overview

This repository demonstrates a Retrieval-Augmented Generation (RAG) pipeline using the `Rag` class and evaluates its performance using the `Ragas` framework. The main components are:

- `chat.py`: Processes files, embeds text, and handles user queries.
- `eval.py`: Evaluates the RAG pipeline using `Ragas`.

## Prerequisites

Ensure you have the required packages installed:

```sh
pip install -r requirements.txt
```

## Running the Chat

The `chat.py` script processes text and PDF files, embeds the text, and allows user interaction for querying the embedded data.

### Steps:

1. Place your text and PDF files in the `data` folder.
2. Run the script:

   ```sh
   python -m app.chat.py
   ```

3. Follow the prompts to add URLs, search, or exit.

## Evaluating the RAG Pipeline

The `eval.py` script processes files, generates a test set, and evaluates the RAG pipeline using `Ragas`.

### Steps:

1. Ensure your data files are in the `data` folder.
2. Run the script:

   ```sh
   python -m app.eval.py
   ```

3. The script will generate and save evaluation results in `evaluation_results.csv`.

## Key Files

- `rag.py`: Contains the `Rag` class with methods for processing and embedding text.

The rag file is the most important file in this project. It contains the `Rag` class with methods for processing and embedding text and is used by the chat, eval, and Streamlit files (and will be used by the FastAPI implementation to come). The functions in this Rag were primarily authored by the marvelous echohive [echohive | Building AI powered apps | Patreon](https://www.patreon.com/echohive42/posts). Echo also conceived the simple chat loop for quick RAG testing.

- `chat.py`: Main chat loop for processing and querying data.
- `eval.py`: Script for evaluating the RAG pipeline.

The RAGAS set up was inspired by by
Coding-Crashkurse [Coding Crashkurse - YouTube](https://www.youtube.com/@codingcrashkurse6429).

- `streamlit.py`: Streamlit app for interacting with the RAG pipeline and RAGAS evaluation.

- `requirements.txt`: Lists the required Python packages.

## Example Usage

### Running the Chat

```sh
python -m app.chat.py
```

### Running the Evaluation

```sh
python -m app.eval.py
```

### Running Streamlit

```sh
streamlit run app/streamlit.py
```

## Additional Information

For more details on `Ragas`, refer to the [Ragas Documentation](https://docs.ragas.io/en/stable/).
