import streamlit as st
import os
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def get_pdf_text(filename):
    return extract_text(filename)

class CustomVectorStore:
    def __init__(self, texts):
        self.vectorizer = TfidfVectorizer()
        self.vectors = self.vectorizer.fit_transform(texts)
        self.texts = texts

    def similarity_search(self, query, k=2):
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.vectors)[0]
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return [self.texts[i] for i in top_k_indices]

def process_uploaded_files(uploaded_files):
    vectorstores = {}
    raw_texts = {}
    for uploaded_file in uploaded_files:
        if uploaded_file.type == 'application/pdf':
            raw_text = get_pdf_text(uploaded_file)
            texts = raw_text.split('\n\n')  # Simple text splitting
            vectorstore = CustomVectorStore(texts)
            vectorstores[uploaded_file.name] = vectorstore
            raw_texts[uploaded_file.name] = raw_text
    return vectorstores, raw_texts

def generate_response(query_text, vectorstores):
    docs = ""
    for file_name, vectorstore in vectorstores.items():
        relevant_texts = vectorstore.similarity_search(query_text, k=2)
        for i, text in enumerate(relevant_texts):
            docs += f"'{file_name}-문서{i+1}':{text}\n"
    
    messages = [
        {"role": "system", "content": "너는 여러 문서에 대해 질의응답을 하는 '리냥이'야. 주어진 논문들과 문서들을 참고하여 사용자의 질문에 답변을 해줘. 문서에 내용이 부족하다면 네가 알고 있는 지식을 포함해서 답변해줘"},
        {"role": "user", "content": f"질문:{query_text}\n\n{docs}"}
    ]
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0,
    )
    
    return response.choices[0].message['content']

def generate_summarize(raw_text):
    messages = [
        {"role": "system", "content": "Summarize the document in 'Notion style'. After briefly summarizing the Introduction, explain Method, Result, and Discussion in as much detail as possible using bullet points for each chapter. Excluding References content"},
        {"role": "user", "content": raw_text}
    ]
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0,
    )
    
    return response.choices[0].message['content']

def analyze_keyword(raw_text, keyword):
    messages = [
        {"role": "system", "content": f"다음 나올 문서에 {keyword}와 관련된 내용이 있는지 분석해줘."},
        {"role": "user", "content": raw_text}
    ]
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0,
    )
    
    return response.choices[0].message['content']

def abstract_summary(raw_text):
    messages = [
        {"role": "system", "content": "Read and analyze the document and write a report of approximately 3000 words to the professor."},
        {"role": "user", "content": raw_text}
    ]
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0,
    )
    
    return response.choices[0].message['content']

def compare_documents(raw_texts):
    content = "\n\n".join([f"{file_name}:\n{raw_text}" for file_name, raw_text in raw_texts.items()])
    
    messages = [
        {"role": "system", "content": "여러 문서의 내용을 비교 분석해주세요. 각 문서의 주요 주제, 방법론, 결과, 그리고 결론을 비교하고, 문서들 간의 유사점과 차이점을 강조해주세요."},
        {"role": "user", "content": content}
    ]
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0,
    )
    
    return response.choices[0].message['content']

# Streamlit UI code remains largely unchanged
st.set_page_config(page_title='/ᐠ ._. ᐟ\ﾉ 다중 문서 기반 요약 및 QA 챗봇')
st.title('/ᐠ ._. ᐟ\ﾉ The leelab \n 다중 문서 기반 요약 및 QA 챗봇')

api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
save_button = st.sidebar.button("Save Key")
if save_button and len(api_key)>10:
    os.environ["OPENAI_API_KEY"] = api_key
    openai.api_key = api_key
    st.sidebar.success("API Key saved successfully!")

keyword = st.sidebar.text_input("Enter keyword to analyze", value="")

uploaded_files = st.file_uploader('Upload PDF documents', type=['pdf'], accept_multiple_files=True)

if uploaded_files:
    vectorstores, raw_texts = process_uploaded_files(uploaded_files)
    st.session_state['vectorstores'] = vectorstores
    st.session_state['raw_texts'] = raw_texts

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "항상 수고가 많으십니다!≽^•⩊•^≼ 어떤게 궁금하신가요? =^._.^= ∫"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("'Sum', 'Keyword', 'Report', 'Compare' 또는 질문을 입력해주세요 /ᐠ •ヮ• マ Ⳋ"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        if prompt == "Sum":
            for file_name, raw_text in st.session_state['raw_texts'].items():
                response = generate_summarize(raw_text)
                st.write(f"Summary for {file_name}:")
                st.write(response)
                st.write("---")
        elif prompt == "Keyword":
            for file_name, raw_text in st.session_state['raw_texts'].items():
                response = analyze_keyword(raw_text, keyword)
                st.write(f"Keyword analysis for {file_name}:")
                st.write(response)
                st.write("---")
        elif prompt == "Report":
            for file_name, raw_text in st.session_state['raw_texts'].items():
                response = abstract_summary(raw_text)
                st.write(f"Report for {file_name}:")
                st.write(response)
                st.write("---")
        elif prompt == "Compare":
            response = compare_documents(st.session_state['raw_texts'])
            st.write(response)
        else:
            response = generate_response(prompt, st.session_state['vectorstores'])
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
