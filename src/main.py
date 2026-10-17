import os
import uuid
import subprocess

from typing import TypedDict, Annotated, Literal
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import tool
from langgraph.types import interrupt, Command

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b")


@tool(description="Writes or overwrites text content to a specific file inside the workspace directory.")
def write_file(filename: str, content: str) -> str:
    workspace = Path(__file__).resolve().parent / "workspace"
    workspace.mkdir(exist_ok=True)
    target = workspace / filename
    target.write_text(content, encoding="utf-8")
    return f"File {filename} written successfully."


llm = llm.bind_tools([write_file])

KNOWLEDGE = [
    "RAG (Retrieval-Augmented Generation) is a technique that combines an external search/database retrieval step with an LLM to answer questions using private or up-to-date factual data. It was made by David :p",
    "LangGraph is a Python framework built on top of LangChain for creating stateful, multi-actor, and cyclical agent workflows using graph structures.",
    "Groq is an AI infrastructure company that builds LPUs (Language Processing Units) designed to run open-source models like Llama at ultra-fast inference speeds.",
    "Greenwashing refers to misleading claims made by companies to make their products or policies seem far more environmentally friendly than they actually are.",
    "Wikipedia was launched in 2001 by Jimmy Wales and Larry Sanger as a free, openly editable multilingual online encyclopedia."
]

vector_store = InMemoryVectorStore(FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5"))
vector_store.add_documents([Document(page_content=text) for text in KNOWLEDGE])


class IntentClassifier(BaseModel):
    message_intent: Literal['chat', 'knowledge', 'code'] = Field(
        description="Classify whether the user wants to just chat ('chat'), ask for knowledge or facts ('knowledge') or change code ('code')."
    )


class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_intent: str | None

    next_node: str | None


def prepare_coding_request(state: State):
    messages = [{'role': 'system', 'content': 'Rewrite the lastest user request into clear instructions for Groq. U'
                                              'se the conversation history as context. Output only instruction, no explanation'}] + \
               state['messages']

    response = llm.invoke(messages)

    return {'messages': [{'role': 'user', 'content': response.content}]}

def accept_coding(state: State):
    user_prompt = state['messages'][-1].content
    decision = interrupt(f"About to run Groq Code with request: \n\n{user_prompt}\n\n Approve?")

    text = str(decision).strip().lower()

    if text in ['y', 'yes', 'approve', 'ok']:
        return {}

    if text in ['n', 'no']:
        return {'messages': [{'role': 'assistant', 'content': 'Coding request denied by the user'}],
                'next_node': 'denied'}

    return {'messages': [{'role': 'user', 'content': text}], 'next_node': 'denied'}


def classify_intent(state: State):
    structured_llm = llm.with_structured_output(IntentClassifier, method="json_mode")

    system_prompt = (
        "You are an intent classifier. Your ONLY job is to classify the intent of the user. "
        "Do NOT answer the question. Choose one of: 'chat', 'knowledge', 'code'."
        "Return your response strictly as a JSON object with the field 'message_intent'. "
    )

    result = structured_llm.invoke([
                                       {'role': 'system', 'content': system_prompt}
                                   ] + state['messages'])

    return {'message_intent': result.message_intent}


def prompt_llm_chat(state: State):
    messages = [{'role': 'system', 'content': 'You are a friendly chatbot. Be nice.'}] + state['messages']
    response = llm.invoke(messages)
    return {'messages': [response]}


def prompt_llm_rag(state: State):
    query = state['messages'][-1].content
    documents = vector_store.similarity_search(query, k=2)

    context = '\n'.join(f'- {doc.page_content}' for doc in documents)

    system_prompt = (
        f"You are a RAG agent. Answer the user question strictly using ONLY the context provided below. "
        f"If the answer is not present in the context, say 'I don't know'.\n\n"
        f"Context:\n{context}"
    )

    messages = [{'role': 'system', 'content': system_prompt}] + state['messages']
    response = llm.invoke(messages)
    return {'messages': [response]}


import os
from pathlib import Path


def prompt_llm_code(state: State):
    workspace_dir = Path(__file__).resolve().parent / "workspace"
    workspace_dir.mkdir(exist_ok=True)

    system_prompt = (
        "You are an expert coding assistant. Generate clean, functional code. "
        "At the end of your explanation, provide the complete code."
        "Do not output markdown code blocks or explanations when creating files, just call the tools."
    )

    messages = [
                   {'role': 'system', 'content': system_prompt}
               ] + state['messages']

    response = llm.invoke(messages)

    if response.tool_calls:
        print(response.tool_calls)
        tool_results = []
        for call in response.tool_calls:
            if call["name"] == "write_file":
                res = write_file.invoke(call["args"])
                tool_results.append(res)

        reply_text = "\n".join(tool_results)
        return {'messages': [{'role': 'assistant', 'content': reply_text}]}

    return {'messages': [response]}


graph_builder = StateGraph(State)
graph_builder.add_node('classifier', classify_intent)
graph_builder.add_node('chat_agent', prompt_llm_chat)
graph_builder.add_node('rag_agent', prompt_llm_rag)
graph_builder.add_node('code_agent', prompt_llm_code)
graph_builder.add_node('prepare_coding_request', prepare_coding_request)
graph_builder.add_node('accept_coding', accept_coding)

graph_builder.add_edge(START, 'classifier')
graph_builder.add_conditional_edges(
    'classifier',
    lambda state: state['message_intent'],
    {
        'chat': 'chat_agent',
        'knowledge': 'rag_agent',
        'code': 'prepare_coding_request',
    }
)
graph_builder.add_edge('prepare_coding_request', 'accept_coding')
graph_builder.add_conditional_edges('accept_coding',
                                    lambda state: END if state.get('next_node') == 'denied' else 'code_agent')

graph_builder.add_edge('chat_agent', END)
graph_builder.add_edge('rag_agent', END)
graph_builder.add_edge('code_agent', END)

checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

graph.get_graph().draw_mermaid_png(output_file_path='graph.png')

config = {'configurable': {'thread_id': str(uuid.uuid4())}}

if __name__ == '__main__':

    while True:
        user_input = input('Enter a message: ')
        result = graph.invoke({'messages': [{'role': 'user', 'content': user_input}]}, config=config)

        while '__interrupt__' in result:
            prompt = result['__interrupt__'][0].value
            decision = input(f'{prompt}\n>')

            result = graph.invoke(Command(resume=decision), config=config)

        print(f"\n[Agent]: {result['messages'][-1].content}\n")
