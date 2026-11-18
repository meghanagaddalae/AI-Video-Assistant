
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError(
        "GOOGLE_API_KEY not found. Check your .env file."
    )

print("API key loaded: True")
print("Testing Gemini API...")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=api_key,
    temperature=0.3
)

try:
    response = llm.invoke(
        "Give me a short title for a video about artificial intelligence."
    )

    print("\nGemini response:")
    for item in response.content:
        if item.get("type") == "text":
            print(item["text"])
except Exception as e:
    print("\nGemini request failed:")
    print(type(e).__name__)
    print(str(e))
