import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import ChatMessage
from dotenv import load_dotenv
import olefile
import zlib
import struct
from pdfminer.high_level import extract_text
import os

load_dotenv()

# Existing StreamHandler class remains unchanged

# Existing get_hwp_text and get_pdf_text functions remain unchanged

def process_uploaded_files(uploaded_files):
    vectorstores = {}
    raw_texts = {}
    for uploaded_file in uploaded_files:
        if uploaded_file.type == 'application/pdf':
            raw_text = get_pdf_text(uploaded_file)
        elif uploaded_file.type == 'application/octet-stream':
            raw_text = get_hwp_text(uploaded_file)
        else:
            st.warning(f"Unsupported file type: {uploaded_file.type}")
            continue

        text_splitter = CharacterTextSplitter(
            separator = "\n\n",
            chunk_size = 1000,
            chunk_overlap  = 200,
            length_function = len,
            is_separator_regex = False,
        )
        all_splits = text_splitter.create_documents([raw_text])
        
        vectorstore = FAISS.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
        
        vectorstores[uploaded_file.name] = vectorstore
        raw_texts[uploaded_file.name] = raw_text

    return vectorstores, raw_texts

def generate_response(query_text, vectorstores, callback):
    llm = ChatOpenAI(model_name="gpt-4", temperature=0, streaming=True, callbacks=[callback])
    
    all_docs = ""
    for file_name, vectorstore in vectorstores.items():
        docs_list = vectorstore.similarity_search(query_text, k=2)
        for i, doc in enumerate(docs_list):
            all_docs += f"'{file_name}-문서{i+1}':{doc.page_content}\n"

    rag_prompt = [
        SystemMessage(
            content="너는 여러 문서에 대해 질의응답을 하는 '리냥이'야. 주어진 논문들과 문서들을 참고하여 사용자의 질문에 답변을 해줘. 문서에 내용이 부족하다면 네가 알고 있는 지식을 포함해서 답변해줘"
        ),
        HumanMessage(
            content=f"질문:{query_text}\n\n{all_docs}"
        ),
    ]
    
    response = llm(rag_prompt)
    return response.content

def generate_summarize(raw_texts, callback):
    llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0, streaming=True, callbacks=[callback])

    summaries = {}
    for file_name, raw_text in raw_texts.items():
        rag_prompt = [
            SystemMessage(
                content="Summarize the document in 'Notion style'. After briefly summarizing the Introduction, explain Method, Result, and Discussion in as much detail as possible using bullet points for each chapter. Excluding References content"
            ),
            HumanMessage(
                content=raw_text
            ),
        ]
        response = llm(rag_prompt)
        summaries[file_name] = response.content

    return summaries

def analyze_keyword(raw_texts, callback, keyword):
    llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0, streaming=True, callbacks=[callback])

    analyses = {}
    for file_name, raw_text in raw_texts.items():
        rag_prompt = [
            SystemMessage(
                content=f"다음 나올 문서에 {keyword}와 관련된 내용이 있는지 분석해줘."
            ),
            HumanMessage(
                content=raw_text
            ),
        ]
        response = llm(rag_prompt)
        analyses[file_name] = response.content

    return analyses

def compare_documents(raw_texts, callback):
    llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0, streaming=True, callbacks=[callback])

    rag_prompt = [
        SystemMessage(
            content="여러 문서의 내용을 비교 분석해주세요. 각 문서의 주요 주제, 방법론, 결과, 그리고 결론을 비교하고, 문서들 간의 유사점과 차이점을 강조해주세요."
        ),
        HumanMessage(
            content="\n\n".join([f"{file_name}:\n{raw_text}" for file_name, raw_text in raw_texts.items()])
        ),
    ]

    response = llm(rag_prompt)
    return response.content

# Streamlit UI
st.set_page_config(page_title='/ᐠ ._. ᐟ\ﾉ 다중 문서 기반 요약 및 QA 챗봇')
st.title('/ᐠ ._. ᐟ\ﾉ The leelab \n 다중 문서 기반 요약 및 QA 챗봇')

# API Key input
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
save_button = st.sidebar.button("Save Key")
if save_button and len(api_key)>10:
    os.environ["OPENAI_API_KEY"] = api_key
    st.sidebar.success("API Key saved successfully!")

keyword = st.sidebar.text_input("Enter keyword to analyze", value="")

# Multiple file upload
uploaded_files = st.file_uploader('Upload documents', type=['hwp','pdf'], accept_multiple_files=True)

if uploaded_files:
    vectorstores, raw_texts = process_uploaded_files(uploaded_files)
    if vectorstores:
        st.session_state['vectorstores'] = vectorstores
        st.session_state['raw_texts'] = raw_texts

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        ChatMessage(
            role="assistant", content="항상 수고가 많으십니다!≽^•⩊•^≼ 어떤게 궁금하신가요? =^._.^= ∫"
        )
    ]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

# Chat input
if prompt := st.chat_input("'Sum', 'Keyword', 'Compare', 또는 질문을 입력해주세요 /ᐠ •ヮ• マ Ⳋ"):
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        
        if prompt == "Sum":
            summaries = generate_summarize(st.session_state['raw_texts'], stream_handler)
            response = "\n\n".join([f"{file_name}:\n{summary}" for file_name, summary in summaries.items()])
        elif prompt == "Keyword":
            analyses = analyze_keyword(st.session_state['raw_texts'], stream_handler, keyword)
            response = "\n\n".join([f"{file_name}:\n{analysis}" for file_name, analysis in analyses.items()])
        elif prompt == "Compare":
            response = compare_documents(st.session_state['raw_texts'], stream_handler)
        else:
            response = generate_response(prompt, st.session_state['vectorstores'], stream_handler)

        st.session_state["messages"].append(
            ChatMessage(role="assistant", content=response)
        )
