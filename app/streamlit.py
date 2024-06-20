import streamlit as st
from streamlit_chat import message
import asyncio
import os
import sys

# Add the root directory to the sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

import pandas as pd
from app.rag import Rag
from app.eval import main as ragas_eval_main

# Set page title
st.set_page_config(page_title="RAG with RAGAS Evaluation", page_icon=":robot_face:")

# Initialize the Rag instance
rag = Rag()

# Sidebar
st.sidebar.title("Application Settings")
openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
uploaded_files = st.sidebar.file_uploader(
    "Upload data files", type=["doc", "pdf", "txt"], accept_multiple_files=True
)

# Add a text input for URLs
url_input = st.sidebar.text_input("Enter a URL")
registered_urls = st.sidebar.empty()

# Initialize session state variables
if "urls" not in st.session_state:
    st.session_state.urls = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

if "query" not in st.session_state:
    st.session_state.query = ""

if "top_k" not in st.session_state:
    st.session_state.top_k = 1

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "top_chunks" not in st.session_state:
    st.session_state.top_chunks = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join("data", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
    st.sidebar.success("Files uploaded successfully!")
    st.session_state.uploaded_files = uploaded_files

if url_input:
    st.session_state.urls.append(url_input)
    registered_urls.text("Registered URLs:\n" + "\n".join(st.session_state.urls))

    async def process_url():
        with st.spinner("Processing URL..."):
            all_chunks = []
            embedded_chunks = []
            text = await rag.fetch_text_from_url(url_input)
            chunks = rag.process_text(text)
            all_chunks.extend(chunks)
            rag.save_chunks_to_file(all_chunks, filename="app/output/data_chunks.json")
            new_embedded_chunks = await rag.embed_text_chunks(all_chunks)
            embedded_chunks.extend(new_embedded_chunks)
            rag.save_chunks_to_file(
                embedded_chunks, filename="app/output/embeddings.json"
            )
        st.sidebar.success("URL processed successfully!")

    asyncio.create_task(process_url())

# Main section
st.title("RAG with RAGAS Evaluation")

# Create tabs
tabs = st.tabs(["RAG", "RAGAS"])

# RAG tab
with tabs[0]:
    st.subheader("Chat with the Bot")

    query_input = st.text_input("Enter your query", value=st.session_state.query)
    top_k_input = st.number_input(
        "Enter the number of top results to retrieve",
        min_value=1,
        step=1,
        value=st.session_state.top_k,
    )

    if st.button("Submit"):
        st.session_state.query = query_input
        st.session_state.top_k = top_k_input
        st.session_state.top_chunks = []

        async def initialize_chat():
            all_chunks = []
            embedded_chunks = []

            # Process uploaded files
            if st.session_state.uploaded_files:
                for uploaded_file in st.session_state.uploaded_files:
                    file_path = os.path.join("data", uploaded_file.name)
                    if uploaded_file.name.endswith(".txt"):
                        text = rag.load_text_file(file_path)
                    elif uploaded_file.name.endswith(".pdf"):
                        text = rag.load_pdf_file(file_path)
                    else:
                        continue
                    chunks = rag.process_text(text)
                    all_chunks.extend(chunks)

            # Process URLs
            if st.session_state.urls:
                for url in st.session_state.urls:
                    text = await rag.fetch_text_from_url(url)
                    chunks = rag.process_text(text)
                    all_chunks.extend(chunks)

            # Save and embed chunks
            rag.save_chunks_to_file(all_chunks, filename="app/output/data_chunks.json")
            new_embedded_chunks = await rag.embed_text_chunks(all_chunks)
            embedded_chunks.extend(new_embedded_chunks)
            rag.save_chunks_to_file(
                embedded_chunks, filename="app/output/embeddings.json"
            )

            if st.session_state.query and st.session_state.top_k:
                query_embedding = await rag.embed_query(st.session_state.query)
                top_chunks = rag.cosine_similarity_search(
                    query_embedding, embedded_chunks, top_k=st.session_state.top_k
                )

                rag.save_top_chunks_text_to_file(
                    top_chunks, filename="app/output/top_chunks.json"
                )

                # Store top chunks in session state
                st.session_state.top_chunks = top_chunks

                # Pass the top k chunks to GPT-4 to get an answer to the user query
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Here are some documents that may help answer the user query: {top_chunks}. Please provide an answer to the query only based on the documents. If the documents don't contain the answer, say that you don't know.\n\nquery: {st.session_state.query}",
                    },
                ]

                # Create a placeholder for the streaming response
                response_placeholder = st.empty()
                assistant_reply = ""

                # Stream the response
                async for chunk in rag.call_gpt4_with_streaming_for_streamlit(messages):
                    assistant_reply += chunk
                    response_placeholder.markdown(assistant_reply)

        asyncio.run(initialize_chat())
    # Add the expandable JSON viewer for top_chunks
    if st.session_state.top_chunks:
        with st.expander("Retrieved Chunks"):
            st.json(st.session_state.top_chunks)

# RAGAS tab
with tabs[1]:
    st.subheader("RAGAS Evaluation")
    if st.button("Start RAG Evaluation"):
        with st.spinner("Running RAG evaluation..."):
            asyncio.run(ragas_eval_main())
        st.success("RAG evaluation completed!")

        # Display evaluation results
        if os.path.exists("app/output/evaluation_results.csv"):
            st.subheader("Evaluation Results")
            evaluation_results = pd.read_csv("app/output/evaluation_results.csv")
            st.dataframe(evaluation_results)

        if os.path.exists("app/output/generated_dataset.csv"):
            st.subheader("Generated Dataset")
            generated_dataset = pd.read_csv("app/output/generated_dataset.csv")
            st.dataframe(generated_dataset)

        if os.path.exists("app/output/testset.csv"):
            st.subheader("Test Set")
            testset = pd.read_csv("app/output/testset.csv")
            st.dataframe(testset)

# Warning if OpenAI API key is not provided
if not openai_api_key:
    st.warning("Please enter your OpenAI API key in the sidebar.")
