
import ollama

response = ollama.chat(
    model="qwen3:4b",
    messages=[
        {
            "role": "user",
            "content": "What is RAG? Give a short answer."
        }
    ],
    think=False,
    options={
        "num_predict": 256,
        "temperature": 0
    }
)

print("Answer:", response.message.content)
