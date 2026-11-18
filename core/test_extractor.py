
from extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
)

transcript = """
The team discussed the AI Video Assistant project.
The team decided to use ChromaDB to store embeddings.
Sameera will test the summarization module by Friday.
The team still needs to decide how to evaluate answer accuracy.
"""

try:
    print("\n--- ACTION ITEMS ---")
    print(extract_action_items(transcript))

    print("\n--- KEY DECISIONS ---")
    print(extract_key_decisions(transcript))

    print("\n--- OPEN QUESTIONS ---")
    print(extract_questions(transcript))

except Exception as error:
    print("\nTEST FAILED")
    print(type(error).__name__, error)
