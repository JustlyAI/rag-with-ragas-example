import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import sys
import os
import streamlit as st
import asyncio

# Add the root directory to the sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from app.rag import Rag
from app.eval import Eval


class RagWithRagasApp:
    def __init__(self):
        st.set_page_config(
            page_title="RAG with RAGAS Evaluation", page_icon=":robot_face:"
        )

    def sidebar(self):
        st.sidebar.title("Application Settings")
        st.session_state.openai_api_key = st.sidebar.text_input(
            "Enter your OpenAI API Key",
            type="password",
            value=st.session_state.get("openai_api_key", ""),
        )

        if st.session_state.openai_api_key:
            os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key
            # Initialize RAG and RAGAS with the API key
            self.rag = Rag()
            self.eval = Eval()  # Initialize here

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
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
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
                    chunks = self.rag.process_text(text)
                    all_chunks.extend(chunks)
                    self.rag.save_chunks_to_file(
                        all_chunks, filename="app/output/data_chunks.json"
                    )
                    new_embedded_chunks = await self.rag.embed_text_chunks(all_chunks)
                    embedded_chunks.extend(new_embedded_chunks)
                    self.rag.save_chunks_to_file(
                        embedded_chunks, filename="app/output/embeddings.json"
                    )

                asyncio.run(process_url())
                st.sidebar.success("URL processed successfully!")

    def main_section(self):
        st.title("RAG with RAGAS Evaluation")

        tabs = st.tabs(["RAG", "RAGAS"])

        if not st.session_state.openai_api_key:
            st.warning("Please enter your OpenAI API key in the sidebar.")
            return

        with tabs[0]:
            st.subheader("Chat with the Bot")

            query_input = st.text_input(
                "Enter your query", value=st.session_state.get("query", "")
            )
            top_k_input = st.number_input(
                "Enter the number of top results to retrieve",
                min_value=1,
                step=1,
                value=st.session_state.get("top_k", 1),
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
                            chunks = self.rag.process_text(text)
                            all_chunks.extend(chunks)

                    if st.session_state.urls:
                        for url in st.session_state.urls:
                            text = await self.rag.fetch_text_from_url(url)
                            chunks = self.rag.process_text(text)
                            all_chunks.extend(chunks)

                    self.rag.save_chunks_to_file(
                        all_chunks, filename="app/output/data_chunks.json"
                    )
                    new_embedded_chunks = await self.rag.embed_text_chunks(all_chunks)
                    embedded_chunks.extend(new_embedded_chunks)
                    self.rag.save_chunks_to_file(
                        embedded_chunks, filename="app/output/embeddings.json"
                    )

                    if st.session_state.query and st.session_state.top_k:
                        query_embedding = await self.rag.embed_query(
                            st.session_state.query
                        )
                        top_chunks = self.rag.cosine_similarity_search(
                            query_embedding,
                            embedded_chunks,
                            top_k=st.session_state.top_k,
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
                        ) in self.rag.call_gpt4_with_streaming_for_streamlit(messages):
                            assistant_reply += chunk
                            response_placeholder.markdown(assistant_reply)

                asyncio.run(initialize_chat())

            if st.session_state.top_chunks:
                with st.expander("Retrieved Chunks"):
                    st.json(st.session_state.top_chunks)

        with tabs[1]:
            st.subheader("RAGAS Evaluation")
            if st.button("Start RAG Evaluation"):
                with st.spinner("Running RAG evaluation..."):
                    os.environ["OPENAI_API_KEY"] = (
                        st.session_state.openai_api_key
                    )  # Ensure the API key is set
                    asyncio.run(
                        self.eval.run_evaluation()
                    )  # Use the initialized method
                st.success("RAG evaluation completed!")

                if os.path.exists("app/output/evaluation_results.csv"):
                    st.subheader("Evaluation Results")
                    evaluation_results = pd.read_csv(
                        "app/output/evaluation_results.csv"
                    )
                    st.dataframe(evaluation_results)

                    st.subheader("Evaluation Heatmap")
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
                        labels=evaluation_results["question"],
                        rotation=0,
                        fontsize=12,
                    )
                    plt.xticks(fontsize=12)
                    plt.title("RAGAS Evaluation Heatmap", fontsize=16)
                    plt.xlabel("Metrics", fontsize=14)
                    plt.ylabel("Questions", fontsize=14)
                    st.pyplot(plt)

            if os.path.exists("app/output/testset.csv"):
                st.subheader("Test Set")
                testset = pd.read_csv("app/output/testset.csv")
                st.dataframe(testset)

    def render(self):
        if "openai_api_key" not in st.session_state:
            st.session_state.openai_api_key = ""

        if "urls" not in st.session_state:
            st.session_state.urls = []

        if "uploaded_files" not in st.session_state:
            st.session_state.uploaded_files = []

        if "query" not in st.session_state:
            st.session_state.query = ""

        if "top_k" not in st.session_state:
            st.session_state.top_k = 1

        if "top_chunks" not in st.session_state:
            st.session_state.top_chunks = []

        self.sidebar()
        self.main_section()
        st.stop()


if __name__ == "__main__":
    app = RagWithRagasApp()
    app.render()
