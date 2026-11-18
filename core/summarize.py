# #from langchain_mistralai import ChatMistralAI
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# import os 

# #def get_llm():
# #     return ChatMistralAI(model = "mistral-small-latest", mistral_api_key = os.getenv("MISTRAL_API_KEY"),temperature=0.3)

# def get_llm():
#     return ChatGoogleGenerativeAI(
#         model="gemini-3.8-flash",
#         google_api_key=os.getenv("GOOGLE_API_KEY"),
#         temperature=0.3
#     )


# def split_transcript(transcript: str) -> list:
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size = 3000,
#         chunk_overlap = 200
#     )

#     return splitter.split_text(transcript)

# def summarize(transcript : str) -> str:
#     llm = get_llm()

#     map_prompt = ChatPromptTemplate.from_messages(
#         [
#         ("system", "Summarize this portion of a meeting transcript concisely."),
#         ("human", "{text}"),
#     ]
#     )

#     map_chain = map_prompt | llm | StrOutputParser()

#     chunks = split_transcript(transcript)

#     chunk_summaries = [map_chain.invoke({"text" : chunk}) for chunk in chunks]

#     combined = "\n\n".join(chunk_summaries)

#     combined_prompt = ChatPromptTemplate.from_messages(
#         [
#         (
#             "system",
#             "You are an expert meeting summarizer. Combine these partial summaries "
#             "into one final professional meeting summary in bullet points.",
#         ),
#         ("human", "{text}"),
#     ]
#     )

#     combined_chain = (
#         RunnablePassthrough() | RunnableLambda(lambda x:{"text":x}) | combined_prompt | llm | StrOutputParser()
#     )

#     return combined_chain.invoke(combined)

# def generate_title(transcipt : str) -> str:
#     llm = get_llm()

    

#     title_chain = (
#         RunnablePassthrough() | RunnableLambda(lambda x:{"text":x}) | 
#         ChatPromptTemplate.from_messages([
#              (
#                 "system",
#                 "Based on the meeting transcript, generate a short professional meeting title "
#                 "(max 8 words). Only return the title, nothing else.",
#             ),
#             ("human", "{text}"),
#         ])
#         | llm
#         |StrOutputParser()
#     )

#     return title_chain.invoke(transcipt[:2000])


#new


import os

import ollama
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


# --------------------------------------------------
# 1. MODEL CONFIGURATION
# --------------------------------------------------

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
OLLAMA_MODEL = "qwen3:4b"


def generate_text(system_prompt: str, user_prompt: str) -> str:
    """
    Generate text using either local Ollama or Gemini.
    Select the provider using LLM_PROVIDER in the .env file.
    """

    if LLM_PROVIDER == "ollama":
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            think=False,
            options={
                "temperature": 0,
                "num_predict": 512,
            },
        )

        answer = response.message.content or ""

    elif LLM_PROVIDER == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is missing from your .env file."
            )

        llm = ChatGoogleGenerativeAI(
            model="gemini-3.8-flash",
            google_api_key=api_key,
            temperature=0.3,
        )

        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )

        answer = response.content

        if not isinstance(answer, str):
            raise RuntimeError(
                "Gemini returned a non-text response."
            )

    else:
        raise ValueError(
            "Invalid LLM_PROVIDER. Choose 'ollama' or 'gemini'."
        )

    answer = answer.strip()

    if not answer:
        raise RuntimeError(
            f"{LLM_PROVIDER} returned an empty response."
        )

    return answer


# --------------------------------------------------
# 2. SPLIT THE TRANSCRIPT INTO CHUNKS
# --------------------------------------------------

def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
    )

    return splitter.split_text(transcript)


# --------------------------------------------------
# 3. SUMMARIZE THE TRANSCRIPT
# --------------------------------------------------

def summarize(transcript: str) -> str:
    if not transcript or not transcript.strip():
        raise ValueError("The transcript is empty.")

    print(f"Generating summary using: {LLM_PROVIDER}")

    chunks = split_transcript(transcript)
    chunk_summaries = []

    # Summarize each transcript chunk separately.
    for i, chunk in enumerate(chunks, start=1):
        print(f"Summarizing chunk {i}/{len(chunks)}...")

        summary = generate_text(
            system_prompt=(
                "You are an expert meeting summarizer. "
                "Summarize the supplied transcript accurately and concisely. "
                "Preserve important facts, decisions, names, and action items. "
                "Do not invent information. "
                "Return only the summary, without explaining your reasoning."
            ),
            user_prompt=chunk,
        )

        chunk_summaries.append(summary)

    # Combine the individual summaries into one final summary.
    combined = "\n\n".join(chunk_summaries)

    final_summary = generate_text(
        system_prompt=(
            "You are an expert meeting summarizer. "
            "Combine the supplied partial summaries into one coherent, "
            "professional meeting summary using clear bullet points. "
            "Remove repetition while preserving important details, "
            "decisions, and action items. Do not invent information. "
            "Return only the final summary."
        ),
        user_prompt=combined,
    )

    print("Summary generation complete.")

    return final_summary


# --------------------------------------------------
# 4. GENERATE A MEETING TITLE
# --------------------------------------------------

def generate_title(transcript: str) -> str:
    if not transcript or not transcript.strip():
        raise ValueError("The transcript is empty.")

    print(f"Generating title using: {LLM_PROVIDER}")

    title = generate_text(
        system_prompt=(
            "You generate short, professional meeting titles. "
            "Return only one title, with a maximum of 8 words. "
            "Do not include quotation marks or explanations."
        ),
        user_prompt=transcript[:2000],
    )

    # Keep the title short if the model returns extra text.
    title = " ".join(title.split())

    return title
