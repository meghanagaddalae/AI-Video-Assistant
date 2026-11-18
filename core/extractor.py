# #Actionableitems , decision , questions 

# from langchain_mistralai import ChatMistralAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough, RunnableLambda
# import os 


# def get_llm():
#     return ChatMistralAI(model = "mistral-small-latest", mistral_api_key = os.getenv("MISTRAL_API_KEY"),temperature=0.2)



# def build_chain(system_prompt : str):
#     llm = get_llm()
#     return (
#         RunnablePassthrough() | RunnableLambda(lambda x : {"text" : x}) |ChatPromptTemplate.from_messages([
#         ("system", system_prompt),
#         ("human","{text}"),
#     ]) | llm |StrOutputParser()
#     )

# def extract_action_items(transcript:str)->str:
#     chain = build_chain(
#          "You are an expert meeting analyst. From the meeting transcript, "
#         "extract all action items. For each provide:\n"
#         "- Task description\n"
#         "- Owner (who is responsible)\n"
#         "- Deadline (if mentioned, else write 'Not specified')\n\n"
#         "Format as a numbered list. If none found say 'No action items found.'"
#     )

#     return chain.invoke(transcript)


# def extract_key_decisions(transcript: str) -> str:
#     chain = build_chain(
#         "You are an expert meeting analyst. From the meeting transcript, "
#         "extract all key decisions made. Format as a numbered list. "
#         "If none found say 'No key decisions found.'"
#     )
#     return chain.invoke(transcript)


# def extract_questions(transcript: str) -> str:
#     chain = build_chain(
#         "From the meeting transcript, extract all unresolved questions "
#         "or topics needing follow-up. Format as a numbered list. "
#         "If none found say 'No open questions found.'"
#     )
#     return chain.invoke(transcript)

#new

import os

import ollama
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
OLLAMA_MODEL = "qwen3:4b"


def generate_text(system_prompt: str, transcript: str) -> str:
    if LLM_PROVIDER == "ollama":
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcript},
            ],
            think=False,
            options={
                "temperature": 0,
                "num_predict": 256,
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
            temperature=0.2,
        )

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=transcript),
        ])

        answer = response.content

        if not isinstance(answer, str):
            raise RuntimeError("Gemini returned a non-text response.")

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


def extract_action_items(transcript: str) -> str:
    if not transcript or not transcript.strip():
        return "No transcript available."

    return generate_text(
        system_prompt=(
            "You are an expert meeting analyst. Extract all action items "
            "from the meeting transcript. For each item provide:\n"
            "- Task description\n"
            "- Owner, if mentioned\n"
            "- Deadline, if mentioned; otherwise write 'Not specified'\n\n"
            "Format as a numbered list. Do not invent details. "
            "If no action items are found, say 'No action items found.'"
        ),
        transcript=transcript,
    )


def extract_key_decisions(transcript: str) -> str:
    if not transcript or not transcript.strip():
        return "No transcript available."

    return generate_text(
        system_prompt=(
            "You are an expert meeting analyst. Extract all key decisions "
            "explicitly made in the meeting transcript. "
            "Format as a numbered list. Do not invent decisions. "
            "If none are found, say 'No key decisions found.'"
        ),
        transcript=transcript,
    )


def extract_questions(transcript: str) -> str:
    if not transcript or not transcript.strip():
        return "No transcript available."

    return generate_text(
        system_prompt=(
            "Extract unresolved questions and topics needing follow-up "
            "from the meeting transcript. Format as a numbered list. "
            "Do not invent questions. "
            "If none are found, say 'No open questions found.'"
        ),
        transcript=transcript,
    )
