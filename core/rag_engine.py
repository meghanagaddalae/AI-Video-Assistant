# import os
# from langchain_mistralai import ChatMistralAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough, RunnableLambda
# from core.vector_store import build_vector_store, load_vector_store, get_retriever

# def get_llm():
#     return ChatMistralAI(
#         model="mistral-small-latest",
#         mistral_api_key=os.getenv("MISTRAL_API_KEY"),
#         temperature=0.3,
#     )

# def format_docs(docs):
#     return "\n\n".join([doc.page_content for doc in docs])

# def build_rag_chain(transcript:str):

#     vector_store = build_vector_store(transcript)

#     retriever = get_retriever(vector_store, k = 4)

#     llm = get_llm()

#     prompt = ChatPromptTemplate.from_messages(

#         [(
#             "system",
#             """You are an expert meeting assistant. Answer the user's question 
# based ONLY on the meeting transcript context provided below.

# If the answer is not found in the context, say: 
# "I could not find this information in the meeting transcript."

# Always be concise and precise. If quoting someone, mention it clearly.

# Context from meeting transcript:
# {context}""",
#         ),
#         ("human", "{question}"),
#     ]
#     )

#     #full LCEL Rag pipeline 

#     rag_chain = (

#         {"context" : retriever | RunnableLambda(format_docs),
#          "question": RunnablePassthrough()
#          }
#          |prompt|llm|StrOutputParser()
#     )

#     return rag_chain


# def load_rag_chain():
#     vector_store = load_vector_store()
#     retriver = get_retriever()

#     llm = get_llm()
#     prompt = ChatPromptTemplate.from_messages([
#         (
#             "system",
#             """You are an expert meeting assistant. Answer the user's question 
# based ONLY on the meeting transcript context provided below.

# If the answer is not found in the context, say: 
# "I could not find this information in the meeting transcript."

# Always be concise and precise. If quoting someone, mention it clearly.

# Context from meeting transcript:
# {context}""",
#         ),
#         ("human", "{question}"),
#     ])

#     rag_chain = (
#         {
#             "context":  retriver| RunnableLambda(format_docs),
#             "question": RunnablePassthrough(),
#         }
#         | prompt
#         | llm
#         | StrOutputParser()
#     )

#     return rag_chain


# def ask_question(rag_chain, question:str) -> str:
#     print(f"Question : {question}")
#     answer = rag_chain.invoke(question)
#     print(f"answer :{answer}")
#     return answer

#new

import os

import ollama
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage

from core.vector_store import (
    build_vector_store,
    load_vector_store,
    get_retriever,
)

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
OLLAMA_MODEL = "qwen3:4b"


def get_llm():
    """Return the configured LLM provider."""
    if LLM_PROVIDER == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is missing from .env")

        return ChatGoogleGenerativeAI(
            model="gemini-3.8-flash",
            google_api_key=api_key,
            temperature=0.2,
        )

    if LLM_PROVIDER == "ollama":
        return None

    raise ValueError("LLM_PROVIDER must be 'ollama' or 'gemini'.")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def generate_answer(question: str, context: str) -> str:
    system_prompt = (
        "You are an expert meeting assistant. Answer the user's question "
        "based ONLY on the meeting transcript context provided below.\n\n"
        "If the answer is not found in the context, say: "
        "'I could not find this information in the meeting transcript.'\n\n"
        "Always be concise and precise. Do not invent facts. "
        "If quoting someone, mention it clearly.\n\n"
        "Context from meeting transcript:\n"
        f"{context}"
    )

    if LLM_PROVIDER == "ollama":
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            think=False,
            options={"temperature": 0, "num_predict": 256},
        )
        answer = response.message.content or ""

    else:
        llm = get_llm()
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=question),
        ])
        answer = response.content

        if not isinstance(answer, str):
            raise RuntimeError("Gemini returned a non-text response.")

    answer = answer.strip()
    if not answer:
        raise RuntimeError(f"{LLM_PROVIDER} returned an empty response.")

    return answer


def build_rag_chain(transcript: str):
    if not transcript or not transcript.strip():
        raise ValueError("The transcript is empty.")

    vector_store = build_vector_store(transcript)
    retriever = get_retriever(vector_store, k=4)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert meeting assistant. Answer the user's question
based ONLY on the meeting transcript context provided below.

If the answer is not found in the context, say:
"I could not find this information in the meeting transcript."

Always be concise and precise. Do not invent facts.
If quoting someone, mention it clearly.

Context from meeting transcript:
{context}""",
        ),
        ("human", "{question}"),
    ])

    if LLM_PROVIDER == "gemini":
        llm = get_llm()
        return (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

    def answer_with_context(inputs):
        context = format_docs(retriever.invoke(inputs["question"]))
        return generate_answer(inputs["question"], context)

    return RunnableLambda(
        lambda question: answer_with_context({"question": question})
    )


def load_rag_chain():
    vector_store = load_vector_store()
    retriever = get_retriever(vector_store, k=4)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert meeting assistant. Answer the user's question
based ONLY on the meeting transcript context provided below.

If the answer is not found in the context, say:
"I could not find this information in the meeting transcript."

Always be concise and precise. Do not invent facts.
If quoting someone, mention it clearly.

Context from meeting transcript:
{context}""",
        ),
        ("human", "{question}"),
    ])

    if LLM_PROVIDER == "gemini":
        llm = get_llm()
        return (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

    def answer_with_context(inputs):
        context = format_docs(retriever.invoke(inputs["question"]))
        return generate_answer(inputs["question"], context)

    return RunnableLambda(
        lambda question: answer_with_context({"question": question})
    )


def ask_question(rag_chain, question: str) -> str:
    print(f"Question: {question}")
    answer = rag_chain.invoke(question)
    print(f"Answer: {answer}")
    return answer
