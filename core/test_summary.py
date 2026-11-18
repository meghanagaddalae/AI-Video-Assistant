
from summarize import summarize, generate_title

sample_transcript = """
The project team discussed the AI Video Assistant.
The system accepts video or audio files and extracts the transcript.
The team decided to use ChromaDB for storing embeddings.
The next task is to test the summarization module.
The development team will complete testing by Friday.
"""

try:
    print("\n--- MEETING SUMMARY ---")
    summary = summarize(sample_transcript)
    print(summary)

    print("\n--- MEETING TITLE ---")
    title = generate_title(sample_transcript)
    print(title)

except Exception as error:
    print("\nTEST FAILED")
    print(type(error).__name__, error)
