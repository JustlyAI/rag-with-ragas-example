import asyncio
import os
from termcolor import colored

from app.rag import Rag


async def chat_loop():
    all_chunks = []
    embedded_chunks = []
    data_folder = "app/data"

    # Create an instance of the Rag class
    rag = Rag()

    # Ask user y/n if you would like to clear existing output
    if (
        input(colored("Would you like to clear existing output? (y/n): ", "blue"))
        == "y"
    ):
        rag.clear_output_folder()

    # Automatically process files from the data folder
    print(colored("Processing files from the data folder", "magenta"))
    for file_name in os.listdir(data_folder):
        file_path = os.path.join(data_folder, file_name)
        if file_name.endswith(".txt"):
            text = rag.load_text_file(file_path)
        elif file_name.endswith(".pdf"):
            text = rag.load_pdf_file(file_path)
        else:
            continue
        chunks = rag.process_text(text)
        all_chunks.extend(chunks)

    number_of_files = len(
        [file for file in os.listdir(data_folder) if file != "__init__.py"]
    )
    print(colored(f"Number of files processed: {number_of_files}.", "yellow"))

    rag.save_chunks_to_file(all_chunks, filename="app/output/data_chunks.json")
    # Exclude __init__.py and count the number of files in the data folder
    print(colored("Chunks saved to data_chunks.json", "green"))
    print(colored(f"Number of chunks processed: {len(all_chunks)}.", "yellow"))
    new_embedded_chunks = await rag.embed_text_chunks(all_chunks)
    embedded_chunks.extend(new_embedded_chunks)
    rag.save_chunks_to_file(embedded_chunks, filename="app/output/embeddings.json")
    print(colored("Chunks and embeddings saved to embeddings.json", "green"))

    while True:
        input_type = input(
            colored(
                "Enter 'urls' to add URLs, 'search' to switch to search mode, or 'exit' to quit: ",
                "blue",
            )
        ).lower()
        if input_type == "exit":
            break
        elif input_type == "search":
            while True:
                query = input(
                    colored(
                        '\nEnter your search query (or type "back" to return): ', "blue"
                    )
                )
                if query.lower() == "back":
                    break
                query_embedding = await rag.embed_query(query)
                top_k = int(
                    input(colored("Enter the number of top results to save: ", "blue"))
                )
                top_chunks = rag.cosine_similarity_search(
                    query_embedding, embedded_chunks, top_k=top_k
                )

                rag.save_top_chunks_text_to_file(
                    top_chunks, filename="app/output/top_chunks.json"
                )
                print(
                    colored(
                        f"Top {top_k} chunks' text saved to top_chunks.json",
                        "green",
                    )
                )

                # Pass the top k chunks to GPT-4o to get an answer to the user query
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Here are some documents that may help answer the user query: {top_chunks}. Please provide an answer to the query only based on the documents. if the documents dont contain the answer, say that you dont know.\n\nquery: {query}",
                    },
                ]
                # print the number of tokens in the messages
                print(
                    colored(
                        f"Number of tokens in the messages: {rag.count_tokens(str(messages))}",
                        "yellow",
                    )
                )
                gpt4_response = await rag.call_gpt4_with_streaming(messages)
        elif input_type == "urls":
            try:
                urls = input(
                    colored("Enter the URLs separated by commas: ", "blue")
                ).split(",")
                for url in urls:
                    text = await rag.fetch_text_from_url(url.strip())
                    chunks = rag.process_text(text)
                    all_chunks.extend(chunks)
                rag.save_chunks_to_file(
                    all_chunks, filename="app/output/data_chunks.json"
                )
                print(colored("Chunks saved to data_chunks.json", "green"))
                new_embedded_chunks = await rag.embed_text_chunks(all_chunks)
                embedded_chunks.extend(new_embedded_chunks)
                rag.save_chunks_to_file(
                    embedded_chunks, filename="app/output/embeddings.json"
                )
                print(
                    colored("Chunks and embeddings saved to embeddings.json", "green")
                )
            except Exception as e:
                print(colored(f"An error occurred: {e}", "red"))
        else:
            print(
                colored(
                    'Invalid input. Please enter "urls", "search", or "exit".', "red"
                )
            )


if __name__ == "__main__":
    asyncio.run(chat_loop())
