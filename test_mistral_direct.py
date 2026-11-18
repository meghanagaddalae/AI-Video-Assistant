import os
from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")

print("API key loaded:", bool(api_key))

client = Mistral(api_key=api_key)

response = client.chat.complete(
    model="mistral-small-2603",
    messages=[
        {
            "role": "user",
            "content": "Give me a short title for a video about artificial intelligence."
        }
    ]
)

print(response.choices[0].message.content)