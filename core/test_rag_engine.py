
from rag_engine import build_rag_chain, ask_question

sample_transcript = """
The team decided to use ChromaDB to store transcript embeddings.
Sameera will test the summarization module by Friday.
The team needs to decide how to evaluate answer accuracy.
"""

try:
    print("Building RAG chain...")
    rag_chain = build_rag_chain(sample_transcript)

    print("\n--- TEST 1 ---")
    ask_question(rag_chain, "What database did the team choose?")

    print("\n--- TEST 2 ---")
    ask_question(rag_chain, "Who will test the summarization module?")

    print("\n--- TEST 3 ---")
    ask_question(rag_chain, "What is the meeting location?")

except Exception as error:
    print("\nTEST FAILED")
    print(type(error).__name__, error)
