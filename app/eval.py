import os
import json
import asyncio

# import seaborn as sns
# import matplotlib.pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap
from datasets import Dataset
from termcolor import colored

from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_relevancy,
    context_recall,
    context_precision,
)

from app.rag import Rag

rag = Rag()


async def main():
    data_folder_path = "data"

    print(colored("Starting to process files in the data folder...", "yellow"))

    # Process all text and PDF files in the data folder, transform them into chunks, and store them
    all_chunks = await rag.process_files_in_folder_for_eval(data_folder_path)

    print(colored("All chunks have been processed and saved.", "blue"))

    print(colored("Starting to generate a test set.", "green"))

    generator = TestsetGenerator.with_openai()

    testset = generator.generate_with_langchain_docs(
        documents=all_chunks,
        test_size=2,
        distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25},
    )
    # Convert the testset to a pandas DataFrame and save it to a CSV file
    testset_df = testset.to_pandas()
    testset_df.to_csv("app/output/testset.csv", index=False)
    print(colored("Test set has been exported to app/output/testset.csv.", "blue"))

    questions = testset.to_pandas()["question"].to_list()
    ground_truth = testset.to_pandas()["ground_truth"].to_list()

    data = {"question": [], "answer": [], "contexts": [], "ground_truth": ground_truth}

    for query in questions:
        data["question"].append(query)

        # Embed the query
        query_embedding = await rag.embed_query(query)

        # Extract embeddings from Document objects
        embedded_chunks = [
            {"text": chunk.page_content, "embedding": chunk.metadata["embedding"]}
            for chunk in all_chunks
        ]

        # Retrieve relevant documents
        top_chunks = rag.cosine_similarity_search(
            query_embedding, embedded_chunks, top_k=3
        )

        rag.save_top_chunks_text_to_file(
            top_chunks, filename="app/output/top_chunks.json"
        )

        # Generate the answer using GPT-4o
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"Here are some documents that may help answer the user query: {top_chunks}. Please provide an answer to the query only based on the documents. If the documents don't contain the answer, say that you don't know.\n\nquery: {query}",
            },
        ]
        answer = await rag.call_gpt4(messages)

        data["answer"].append(answer)
        data["contexts"].append(top_chunks)

    dataset = Dataset.from_dict(data)

    # Export the dataset as a CSV file for inspection
    dataset_df = dataset.to_pandas()
    dataset_df.to_csv("app/output/generated_dataset.csv", index=False)
    print(
        colored(
            "Generated dataset has been exported to app/output/generated_dataset.csv.",
            "blue",
        )
    )

    result = evaluate(
        dataset=dataset,
        metrics=[
            context_relevancy,
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ],
    )

    result_df = result.to_pandas()
    result_df.to_csv("app/output/evaluation_results.csv", index=False)
    print(
        colored(
            "Evaluation results have been exported to app/output/evaluation_results.csv.",
            "blue",
        )
    )


if __name__ == "__main__":
    asyncio.run(main())