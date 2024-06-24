# Basic RAG 📚 with RAGAS Evaluation ✅

Build and evaluate your own Retrieval-Augmented Generation (RAG) system in (almost) plain Python. This is a fully functional and highly modularized RAG system that can be used as a base for building custom RAG systems.

## Introduction

Retrieval Augmented Generation (RAG) is a technique that enhances language models by providing them with relevant information retrieved from a knowledge base. This project demonstrates a RAG pipeline and evaluates its performance using the RAGAS framework.

## Project Structure

- `app/`: Main application directory
  - `data/`: Folder for storing input documents (PDFs, TXTs)
  - `output/`: Folder for storing processed data and evaluation results
  - `rag.py`: Core RAG implementation
  - `chat.py`: Interactive chat interface
  - `eval.py`: RAGAS evaluation script
  - `streamlit.py`: Streamlit web application

## Prerequisites

- IDE (VSCode, PyCharm, Jupyter Notebook, etc.)
- Python 3.10 or later
- Anaconda (recommended)
- Docker (optional)
- OpenAI API Key

## Setting up Conda Environment and Installing Requirements

To set up the Conda environment and install the required packages, follow these steps:

1. **Create a new Conda environment:**

   ```sh
   conda create --name rag_env python=3.10
   ```

2. **Activate the Conda environment:**

   ```sh
   conda activate rag_env
   ```

3. **Install the required packages:**

   ```sh
   pip install -r requirements.txt
   ```

## Setting Up Environment Variables

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

The current implementation defaults to 'gpt-4-turbo' for the LLM model and OpenAI's 'text-embedding-3-large' for the embeddings model.

## Quick Start

**Run these 3 commands to check the application components:**

```sh
python -m app.chat
python -m app.eval
streamlit run app/streamlit.py
```

If everything is set up correctly, you should see the application components running without any errors. Follow Docker Instructions if your system is incompatible.

Coming soon: A deployed basic RAG with RAGAS Streamlit App can be accessed directly at http:// (bring your own key).

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

#### Using Docker Compose (Recommended)

1. **Build and Start the Docker Compose Services:**

   ```sh
   docker compose up --build
   ```

2. **Restart a Previous Build:**

   ```sh
   docker compose up
   ```

3. **End the Docker Compose Session:**

   ```sh
   docker compose down
   ```

## Components

### RAG Pipeline (rag.py)

The `Rag` class in `rag.py` is the core of the RAG system. It handles:

- Text processing and chunking
- Embedding generation
- Similarity search
- Integration with OpenAI's API

### Chat Interface (chat.py)

The chat interface allows you to:

- Process local files (PDFs and TXTs)
- Add URLs for processing
- Perform searches on the embedded data
- Interact with the RAG system
- Currently, the only inputs are query and top-k

Usage:

```sh
python -m app.chat
```

Special attribution for the modular rag with chat is due to the incomparable Mr. echohive [echohive | Building AI powered apps | Patreon](https://www.patreon.com/echohive42/posts).

### RAGAS Evaluation (eval.py)

The evaluation script uses RAGAS to assess the RAG pipeline's performance. It measures:

- Context Relevancy
- Context Precision
- Context Recall
- Faithfulness
- Answer Relevancy

Usage:

```sh
python -m app.eval
```

### Streamlit App (streamlit.py)

The Streamlit app provides a web interface for:

- Uploading documents
- Configuring RAG parameters
- Running queries
- Viewing retrieved contexts
- Performing RAGAS evaluation

Usage:

```sh
streamlit run app/streamlit.py
```

## Customizing the RAG Pipeline

You can customize the RAG pipeline by modifying the following parameters in the Streamlit app or directly in the code:

- Language Model (e.g., gpt-4-turbo, gpt-3.5-turbo)
- Embedding Model (e.g., text-embedding-3-large)
- Chunk Size and Overlap
- Top-K results for retrieval

## Troubleshooting

- **OpenAI API Key Issues**: Ensure your API key is correctly set in the `.env` file or passed as an environment variable.
- **Docker Connection Errors**: Check if the correct ports are exposed and mapped.
- **Out of Memory Errors**: Try reducing the chunk size or the number of documents processed at once.

## Contributing

Contributions to this project are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to your branch
5. Create a pull request

## Acknowledgements

This project has been inspired and built upon the work of brilliant and generous individuals, namely:

- [Modular Rag and chat implementation from URLs, PDFs and txt files. | Patreon](https://www.patreon.com/posts/modular-rag-and-106461497)
- [Coding-Crashkurse/RAG-Evaluation-with-Ragas](https://github.com/Coding-Crashkurse/RAG-Evaluation-with-Ragas)

## Additional Information

For more details on RAGAS, refer to the [RAGAS Documentation](https://docs.ragas.io/en/stable/).
