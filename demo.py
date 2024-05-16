import streamlit as st

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import ChatMessage

from dotenv import load_dotenv

load_dotenv()

# handle streaming conversation
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

# function to extract text from an HWP file
import olefile
import zlib
import struct

def get_hwp_text(filename):
    f = olefile.OleFileIO(filename)
    dirs = f.listdir()

    # HWP 파일 검증
    if ["FileHeader"] not in dirs or \
       ["\x05HwpSummaryInformation"] not in dirs:
        raise Exception("Not Valid HWP.")

    # 문서 포맷 압축 여부 확인
    header = f.openstream("FileHeader")
    header_data = header.read()
    is_compressed = (header_data[36] & 1) == 1

    # Body Sections 불러오기
    nums = []
    for d in dirs:
        if d[0] == "BodyText":
            nums.append(int(d[1][len("Section"):]))
    sections = ["BodyText/Section"+str(x) for x in sorted(nums)]

    # 전체 text 추출
    text = ""
    for section in sections:
        bodytext = f.openstream(section)
        data = bodytext.read()
        if is_compressed:
            unpacked_data = zlib.decompress(data, -15)
        else:
            unpacked_data = data
    
        # 각 Section 내 text 추출    
        section_text = ""
        i = 0
        size = len(unpacked_data)
        while i < size:
            header = struct.unpack_from("<I", unpacked_data, i)[0]
            rec_type = header & 0x3ff
            rec_len = (header >> 20) & 0xfff

            if rec_type in [67]:
                rec_data = unpacked_data[i+4:i+4+rec_len]
                section_text += rec_data.decode('utf-16')
                section_text += "\n"

            i += 4 + rec_len

        text += section_text
        text += "\n"

    return text

# Function to extract text from an PDF file
from pdfminer.high_level import extract_text

def get_pdf_text(filename):
    raw_text = extract_text(filename)
    return raw_text

# document preprocess
def process_uploaded_file(uploaded_file):
    # Load document if file is uploaded
    if uploaded_file is not None:
        # loader
        if uploaded_file.type == 'application/pdf':
            raw_text = get_pdf_text(uploaded_file)
        elif uploaded_file.type == 'application/octet-stream':
            raw_text = get_hwp_text(uploaded_file)

        # splitter
        text_splitter = CharacterTextSplitter(
            separator = "\n\n",
            chunk_size = 1000,
            chunk_overlap  = 200,
            length_function = len,
            is_separator_regex = False,
            )
        all_splits = text_splitter.create_documents([raw_text])

        print("총 " + str(len(all_splits)) + "개의 passage")
        
        # storage
        vectorstore = FAISS.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
                
        return vectorstore, raw_text
    return None

# generate response using RAG technic
def generate_response(query_text, vectorstore, callback):

    # retriever
    docs_list = vectorstore.similarity_search(query_text, k=3)
    docs = ""
    for i, doc in enumerate(docs_list):
        docs += f"'문서{i+1}':{doc.page_content}\n"
        
    # generator
    llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0, streaming=True, callbacks=[callback])
    
    # chaining
    rag_prompt = [
        SystemMessage(
            content="너는 문서에 대해 질의응답을 하는 '리냥이'야. 주어진 논문과 문서를 참고하여 사용자의 질문에 답변을 해줘. 문서에 내용이 부족하다면 네가 알고 있는 지식을 포함해서 답변해줘. 가능한 한 간결하게 답변해."
            ),
        HumanMessage(
            content=f"질문:{query_text}\n\n{docs}"
        ),
    ]
    
    response = llm(rag_prompt)
    
    return response.content


def generate_summarize(raw_text, callback):
    llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0, streaming=True, callbacks=[callback])

    rag_prompt = [
        SystemMessage(
            content="다음 나올 문서를 'Notion style'로 요약해줘. Introduction을 간단하게 요약한 후 Method와 Result 부분은 각 챕터별로 불릿 포인트를 사용해서 최대한 자세하게 설명하도록 해. References 내용은 제외해"
            ),
        HumanMessage(
            content=raw_text
            ),
        ]
    
    response = llm(rag_prompt)
    return response.content

def analyze_keyword(raw_text, callback, keyword):
    # generator
    llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0, streaming=True, callbacks=[callback])

    # prompt formatting
    rag_prompt = [
        SystemMessage(
            content="다음 나올 문서에 " + str(keyword)+"와 관련된 내용이 있는지 분석해줘."
        ),
        HumanMessage(
            content=raw_text
        ),
    ]
    response = llm(rag_prompt)
    return response.content


def abstract_summary(raw_text, callback):
    # generator
    llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0, streaming=True, callbacks=[callback])

    # prompt formatting
    rag_prompt = [
        SystemMessage(
            content="다음 나올 문서를 읽고 분석해서 교수님께 드리는 3000자 내외의 레포트를 작성하도록 해. 가능한 한 간결하게 요약해줘."
        ),
        HumanMessage(
            content=raw_text
        ),
    ]

    
    response = llm(rag_prompt)
    return response.content


# page title
st.set_page_config(page_title='/ᐠ ._. ᐟ\ﾉ 문서 기반 요약 및 QA 챗봇')
st.title('/ᐠ ._. ᐟ\ﾉ The leelab \n 문서 기반 요약 및 QA 챗봇')

# enter token
import os
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
save_button = st.sidebar.button("Save Key")
if save_button and len(api_key)>10:
    os.environ["OPENAI_API_KEY"] = api_key
    st.sidebar.success("API Key saved successfully!")

keyword = st.sidebar.text_input("Enter keyword to analyze", value="")

# file upload
uploaded_file = st.file_uploader('Upload an document', type=['hwp','pdf'])

# file upload logic
if uploaded_file:
    vectorstore, raw_text = process_uploaded_file(uploaded_file)
    if vectorstore:
        st.session_state['vectorstore'] = vectorstore
        st.session_state['raw_text'] = raw_text
        
# chatbot greatings
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        ChatMessage(
            role="assistant", content="항상 수고가 많으십니다!ฅ^•ﻌ•^ฅ 어떤게 궁금하신가요?"
        )
    ]

# conversation history print 
for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)
    
# message interaction
if prompt := st.chat_input("'Sum', 'Keyword', 또는 'Report'를 입력해주세요 =^._.^= ∫"):
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        
        if prompt == "Sum":
            response = generate_summarize(st.session_state['raw_text'],stream_handler)
            st.session_state["messages"].append(
                ChatMessage(role="assistant", content=response)
            )
        elif prompt == "Keyword":
                 response = analyze_keyword(st.session_state['raw_text'], stream_handler, keyword)
                 st.session_state["messages"].append(
                     ChatMessage(role="assistant", content=response)
            )
        elif prompt == "abstract":
            response = abstract_summary(st.session_state['raw_text'], stream_handler)
            st.session_state["messages"].append(
                ChatMessage(role="assistant", content=response)
            )
        else:
            response = generate_response(prompt, st.session_state['vectorstore'], stream_handler)
            st.session_state["messages"].append(
                ChatMessage(role="assistant", content=response)
            )
