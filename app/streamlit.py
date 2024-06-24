import streamlit as st
import pandas as pd
import sys
import os
import asyncio
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Add the root directory to the sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from app.rag import Rag
from app.eval import Eval

from dotenv import load_dotenv

load_dotenv()


class RagWithRagasApp:
    def __init__(self):
        st.set_page_config(
            page_title="RAG with RAGAS Evaluation",
            page_icon=":robot_face:",
            layout="wide",
        )

    def sidebar(self):
        st.sidebar.title("App Settings ⚙️")

        # Check if OpenAI API key is in environment variables
        openai_api_key = os.getenv("OPENAI_API_KEY")

        if openai_api_key:
            st.sidebar.success("OpenAI API key detected.")
            st.session_state.openai_api_key = openai_api_key
        else:
            openai_api_key_input = st.sidebar.text_input(
                "OpenAI API Key",
                type="password",
                placeholder="Paste your API key here (sk-...)",
                help="You can get your API key from https://platform.openai.com/docs/overview.",
                value=st.session_state.get("openai_api_key", ""),
            )

            if openai_api_key_input:
                os.environ["OPENAI_API_KEY"] = openai_api_key_input
                st.session_state.openai_api_key = openai_api_key_input
                st.sidebar.success("OpenAI API key set!")
            else:
                st.sidebar.warning("Please enter your OpenAI API key.")
                st.session_state.openai_api_key = ""

        self.rag = Rag()
        self.eval = Eval()

        st.sidebar.title("Load Data 📤")

        # clear output
        self.rag.clear_output_folder()

        uploaded_files = st.sidebar.file_uploader(
            "Upload data files", type=["doc", "pdf", "txt"], accept_multiple_files=True
        )

        url_input = st.sidebar.text_input("Enter a URL")
        registered_urls = st.sidebar.empty()

        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_path = os.path.join("app/data", uploaded_file.name)
                try:
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                except Exception as e:
                    st.error(f"Error uploading file {uploaded_file.name}: {e}")
            st.sidebar.success("Files uploaded successfully!")
            st.session_state.uploaded_files = uploaded_files

        if url_input:
            st.session_state.urls.append(url_input)
            registered_urls.text(
                "Registered URLs:\n" + "\n".join(st.session_state.urls)
            )

            with st.spinner("Processing URL..."):

                async def process_url():
                    all_chunks = []
                    embedded_chunks = []
                    text = await self.rag.fetch_text_from_url(url_input)
                    chunks = self.rag.process_text(
                        text=text,
                        token_encoding_model=st.session_state.token_encoding_model,
                        chunk_size=st.session_state.chunk_size,
                        overlap=st.session_state.overlap,
                    )
                    all_chunks.extend(chunks)
                    self.rag.save_chunks_to_file(
                        all_chunks, filename="app/output/data_chunks.json"
                    )
                    new_embedded_chunks = await self.rag.embed_text_chunks(
                        chunks=all_chunks,
                        embedding_model=st.session_state.embedding_model.embedding_model,
                    )
                    embedded_chunks.extend(new_embedded_chunks)
                    self.rag.save_chunks_to_file(
                        embedded_chunks, filename="app/output/embeddings.json"
                    )

                asyncio.run(process_url())
                st.sidebar.success("URL processed successfully!")

    def main_section(self):
        st.title("Basic RAG 📚 with RAGAS Evaluation ✅")

        tabs = st.tabs(["RAG", "RAGAS"])

        if not st.session_state.openai_api_key or st.session_state.openai_api_key == "":
            st.warning("Please enter your OpenAI API key in the sidebar.")
            return

        with tabs[0]:
            st.subheader("RAG Configuration")

            with st.form(key="rag_config"):
                col1, col2 = st.columns(2)

                with col1:
                    llm_model_input = st.selectbox(
                        "Language Model",
                        options=["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
                        index=["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"].index(
                            st.session_state.get("llm_model", "gpt-4-turbo")
                        ),
                        help="Large language model for generating responses",
                    )
                    token_encoding_model_input = st.selectbox(
                        "Token Encoding Model",
                        options=["gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
                        index=["gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"].index(
                            st.session_state.get("token_encoding_model", "gpt-4")
                        ),
                        help="Large language model for generating responses",
                    )
                    embedding_model_input = st.selectbox(
                        "Embedding Model",
                        options=[
                            "text-embedding-3-large",
                            "text-embedding-3-small",
                            "text-embedding-ada-002",
                        ],
                        index=[
                            "text-embedding-3-large",
                            "text-embedding-3-small",
                            "text-embedding-ada-002",
                        ].index(
                            st.session_state.get(
                                "embedding_model", "text-embedding-3-large"
                            )
                        ),
                        help="Model used for text embedding",
                    )

                with col2:
                    chunk_size_input = st.slider(
                        "Chunk Size",
                        min_value=100,
                        max_value=2000,
                        value=st.session_state.get("chunk_size", 800),
                        help="Size of text chunks for processing",
                    )
                    overlap_input = st.slider(
                        "Overlap",
                        min_value=0,
                        max_value=2000,
                        value=st.session_state.get("overlap", 400),
                        help="Overlap between text chunks",
                    )
                    top_k_input = st.slider(
                        "Top-K",
                        min_value=1,
                        max_value=10,
                        value=st.session_state.get("top_k", 3),
                        help="Number of top results to retrieve",
                    )

                submit_button = st.form_submit_button(label="Apply Configuration")

                if submit_button:
                    st.success("Configuration updated!")

            st.subheader("Query, Retrieval & Answer")

            query_input = st.text_input(
                "Enter your query", value=st.session_state.get("query", "")
            )

            if st.button("Submit"):
                st.session_state.query = query_input
                st.session_state.top_k = top_k_input
                st.session_state.top_chunks = []

                async def initialize_chat():
                    all_chunks = []
                    embedded_chunks = []

                    if st.session_state.uploaded_files:
                        for uploaded_file in st.session_state.uploaded_files:
                            file_path = os.path.join("app/data", uploaded_file.name)
                            if uploaded_file.name.endswith(".txt"):
                                text = self.rag.load_text_file(file_path)
                            elif uploaded_file.name.endswith(".pdf"):
                                text = self.rag.load_pdf_file(file_path)
                            else:
                                continue
                            chunks = self.rag.process_text(
                                text=text,
                                token_encoding_model=token_encoding_model_input,
                                chunk_size=chunk_size_input,
                                overlap=overlap_input,
                            )
                            all_chunks.extend(chunks)

                    if st.session_state.urls:
                        for url in st.session_state.urls:
                            text = await self.rag.fetch_text_from_url(url)
                            chunks = self.rag.process_text(
                                text=text,
                                token_encoding_model=token_encoding_model_input,
                                chunk_size=chunk_size_input,
                                overlap=overlap_input,
                            )
                            all_chunks.extend(chunks)

                    self.rag.save_chunks_to_file(
                        all_chunks, filename="app/output/data_chunks.json"
                    )
                    new_embedded_chunks = await self.rag.embed_text_chunks(
                        chunks=all_chunks, embedding_model=embedding_model_input
                    )
                    embedded_chunks.extend(new_embedded_chunks)
                    self.rag.save_chunks_to_file(
                        embedded_chunks, filename="app/output/embeddings.json"
                    )

                    if st.session_state.query and st.session_state.top_k:
                        query_embedding = await self.rag.embed_query(
                            query=st.session_state.query,
                            embedding_model=embedding_model_input,
                        )

                        # Check if embedded_chunks is not empty before proceeding
                        if embedded_chunks:
                            top_chunks = self.rag.cosine_similarity_search(
                                query_embedding=query_embedding,
                                embedded_chunks=embedded_chunks,
                                top_k=top_k_input,
                            )

                            self.rag.save_top_chunks_text_to_file(
                                top_chunks, filename="app/output/top_chunks.json"
                            )

                            st.session_state.top_chunks = top_chunks

                            messages = [
                                {
                                    "role": "system",
                                    "content": "You are a helpful assistant.",
                                },
                                {
                                    "role": "user",
                                    "content": f"Here are some documents that may help answer the user query: {top_chunks}. Please provide an answer to the query only based on the documents. If the documents don't contain the answer, say that you don't know.\n\nquery: {st.session_state.query}",
                                },
                            ]

                            response_placeholder = st.empty()
                            assistant_reply = ""

                            async for (
                                chunk
                            ) in self.rag.call_gpt_with_streaming_for_streamlit(
                                messages=messages, model=st.session_state.llm_model
                            ):
                                assistant_reply += chunk
                                response_placeholder.markdown(assistant_reply)
                        else:
                            st.warning(
                                "No embedded chunks found. Please make sure you've uploaded documents or provided URLs."
                            )

                asyncio.run(initialize_chat())

            if st.session_state.top_chunks:
                with st.expander("Retrieved Chunks"):
                    st.json(st.session_state.top_chunks)

        with tabs[1]:
            st.subheader("RAGAS Configuration")

            with st.form(key="ragas_config"):
                # First container with two columns
                container1 = st.container()
                with container1:
                    col1, col2 = st.columns(2)

                    with col1:
                        llm_model_input = st.selectbox(
                            "Language Model",
                            options=["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
                            index=[
                                "gpt-4o",
                                "gpt-4-turbo",
                                "gpt-4",
                                "gpt-3.5-turbo",
                            ].index(st.session_state.get("llm_model", "gpt-4-turbo")),
                            help="Large language model for generating responses",
                        )
                        token_encoding_model_input = st.selectbox(
                            "Token Encoding Model",
                            options=["gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
                            index=[
                                "gpt-4-turbo",
                                "gpt-4",
                                "gpt-3.5-turbo",
                            ].index(
                                st.session_state.get("token_encoding_model", "gpt-4")
                            ),
                            help="Large language model for generating responses",
                        )
                        embedding_model_input = st.selectbox(
                            "Embedding Model",
                            options=[
                                "text-embedding-3-large",
                                "text-embedding-3-small",
                                "text-embedding-ada-002",
                            ],
                            index=[
                                "text-embedding-3-large",
                                "text-embedding-3-small",
                                "text-embedding-ada-002",
                            ].index(
                                st.session_state.get(
                                    "embedding_model", "text-embedding-3-large"
                                )
                            ),
                            help="Model used for text embedding",
                        )

                    with col2:
                        chunk_size_input = st.slider(
                            "Chunk Size",
                            min_value=100,
                            max_value=2000,
                            value=st.session_state.get("chunk_size", 800),
                            help="Size of text chunks for processing",
                        )
                        overlap_input = st.slider(
                            "Overlap",
                            min_value=0,
                            max_value=2000,
                            value=st.session_state.get("overlap", 400),
                            help="Overlap between text chunks",
                        )
                        top_k_input = st.slider(
                            "Top-K",
                            min_value=1,
                            max_value=10,
                            value=st.session_state.get("top_k", 3),
                            help="Number of top results to retrieve",
                        )
                st.divider()
                # Second container with two columns
                container2 = st.container()
                with container2:
                    col1, col2 = st.columns(2)

                    with col1:
                        st.warning(
                            "Increasing the test size significantly increases API costs."
                        )
                        test_size_input = st.slider(
                            "Size of Test Set",
                            min_value=1,
                            max_value=25,
                            value=st.session_state.get("test_size", 10),
                            help="Number of question-answer pairs to include in test set.",
                        )

                    with col2:
                        st.markdown(
                            """
                            ##### Testing Distribution
                            - Simple Retrieval: 50%
                            - Reasoning: 25%
                            - Multi-Context: 25%
                        """
                        )
                        # st.subheader("Distribution (defaults)")
                        # st.write("Simple: 0.5")
                        # st.write("Reasoning: 0.25")
                        # st.write("Multi-Context: 0.25")
                    # Add three separate sliders correspondig to the distributions in session_state.
                    # The three sliders should amount to 1.0 total
                    # A function might be needed to handle the adjustments.

                apply_config_button = st.form_submit_button(label="Apply Configuration")

                if apply_config_button:
                    st.success("Configuration updated!")

            start_evaluation_button = st.button("Start RAG Evaluation")

            if start_evaluation_button:
                with st.spinner("Running RAG evaluation..."):
                    # os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key

                    # Get the test size from session state
                    test_size = test_size_input

                    # Get the distributions from session state
                    distributions = {
                        "simple": st.session_state.distributions["simple"],
                        "reasoning": st.session_state.distributions["reasoning"],
                        "multi_context": st.session_state.distributions[
                            "multi_context"
                        ],
                    }

                    # Run the evaluation with the specified parameters
                    asyncio.run(
                        self.eval.run_evaluation(
                            llm_model=st.session_state.llm_model,
                            token_encoding_model=st.session_state.token_encoding_model,
                            embedding_model=st.session_state.embedding_model,
                            chunk_size=st.session_state.chunk_size,
                            overlap=st.session_state.overlap,
                            top_k=st.session_state.top_k,
                            test_size=st.session_state.test_size,
                            distributions=st.session_state.distributions,
                        )
                    )
                    # return result_df

                st.success("RAG evaluation completed!")

                if os.path.exists("app/output/testset.csv"):
                    st.subheader("Test Set")
                    testset = pd.read_csv("app/output/testset.csv")
                    st.dataframe(testset)

                if os.path.exists("app/output/evaluation_results.csv"):
                    st.subheader("Evaluation Results")
                    evaluation_results = pd.read_csv(
                        "app/output/evaluation_results.csv"
                    )
                    st.dataframe(evaluation_results)

                    # st.subheader("Evaluation Heatmap")
                    heatmap_data = evaluation_results[
                        [
                            "context_relevancy",
                            "context_precision",
                            "context_recall",
                            "faithfulness",
                            "answer_relevancy",
                        ]
                    ]
                    cmap = LinearSegmentedColormap.from_list(
                        "green_red", ["red", "green"]
                    )
                    plt.figure(figsize=(12, 10))
                    sns.heatmap(
                        heatmap_data,
                        annot=True,
                        fmt=".2f",
                        linewidths=0.5,
                        cmap=cmap,
                        annot_kws={"size": 12},
                    )
                    plt.yticks(
                        ticks=range(len(evaluation_results["question"])),
                        labels=[
                            f"Question {i+1}"
                            for i in range(len(evaluation_results["question"]))
                        ],
                        rotation=0,
                        fontsize=12,
                    )
                    plt.xticks(
                        ticks=range(len(heatmap_data.columns)),
                        labels=[
                            "ContextRelevancy",
                            "ContextPrecision",
                            "ContextRecall",
                            "Faithfulness",
                            "AnswerRelevancy",
                        ],
                        fontsize=12,
                    )
                    plt.title("RAGAS Evaluation Heatmap", fontsize=16)
                    plt.xlabel("Metrics", fontsize=14)
                    plt.ylabel("Questions", fontsize=14)
                    st.pyplot(plt.gcf())  # Render the current figure
                    plt.close()  # Close the figure to avoid overlap

    def render(self):
        if "openai_api_key" not in st.session_state:
            st.session_state.openai_api_key = os.getenv("OPENAI_API_KEY", "")

        if "urls" not in st.session_state:
            st.session_state.urls = []

        if "uploaded_files" not in st.session_state:
            st.session_state.uploaded_files = []

        if "query" not in st.session_state:
            st.session_state.query = ""

        if "top_chunks" not in st.session_state:
            st.session_state.top_chunks = []

        if "token_encoding_model" not in st.session_state:
            st.session_state.token_encoding_model = "gpt-4"

        if "embedding_model" not in st.session_state:
            st.session_state.embedding_model = "text-embedding-3-large"

        if "llm_model" not in st.session_state:
            st.session_state.llm_model = "gpt-4-turbo"

        if "chunk_size" not in st.session_state:
            st.session_state.chunk_size = 800

        if "overlap" not in st.session_state:
            st.session_state.overlap = 400

        if "top_k" not in st.session_state:
            st.session_state.top_k = 3

        if "test_size" not in st.session_state:
            st.session_state.test_size = 4

        # Distributions amount to 1.0
        if "distributions" not in st.session_state:
            st.session_state.distributions = {
                "simple": 0.5,
                "reasoning": 0.25,
                "multi_context": 0.25,
            }

        self.sidebar()
        self.main_section()
        st.stop()


if __name__ == "__main__":
    app = RagWithRagasApp()
    app.render()
