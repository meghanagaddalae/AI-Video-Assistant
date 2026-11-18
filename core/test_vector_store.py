
from vector_store import build_vector_store, get_retriever

sample_transcript = """
The team decided to use ChromaDB for storing transcript embeddings.
Sameera will test the summarization module by Friday.
The team needs to decide how to evaluate answer accuracy.
"""

try:
    print("Building vector store...")

    vector_store = build_vector_store(sample_transcript)

    print("\nSearching for relevant transcript chunks...")

    retriever = get_retriever(vector_store, k=2)
    results = retriever.invoke("What is the team's next task?")

    for i, document in enumerate(results, start=1):
        print(f"\nResult {i}:")
        print(document.page_content)

    print("\nVector store test completed.")

except Exception as error:
    print("\nTEST FAILED")
    print(type(error).__name__, error)
