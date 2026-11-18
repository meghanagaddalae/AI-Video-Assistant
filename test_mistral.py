from dotenv import load_dotenv
import os
from langchain_mistralai import ChatMistralAI

load_dotenv()

print("API key loaded:", bool(os.getenv("MISTRAL_API_KEY")))

llm = ChatMistralAI(
    model="mistral-small-latest",
    mistral_api_key=os.getenv("MISTRAL_API_KEY"),
    temperature=0.3
)

response = llm.invoke(
    "Give me a short title for a video about artificial intelligence."
)

print("Mistral response:")
print(response.content)