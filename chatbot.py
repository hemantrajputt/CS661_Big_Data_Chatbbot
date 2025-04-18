import os
import streamlit as st

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

DB_FAISS_PATH = "VectorEmbeddings/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def load_llm(huggingface_repo_id, HF_TOKEN):
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.7, task="text-generation",
        model_kwargs={"token":HF_TOKEN,
                      "max_length":"512"}
    )
    return llm

def main():
    st.title("BigBot: CS661 Course Chatbot!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt=st.chat_input("What's on your mind? Let's explore this course together!")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
                Use the pieces of information provided in the context to answer user's question.
                If you dont know the answer, just say that you dont know, dont try to make up an answer. 
                Dont provide anything out of the given context

                Context: {context}
                Question: {question}

                Start the answer directly. No small talk please.
                """
        
        HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN=os.environ.get("HF_TOKEN")

        try: 
            vectorstore=get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            qa_chain=RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response=qa_chain.invoke({'query':prompt})

            result=response["result"]
            source_documents=response["source_documents"]
            def format_sources(sources):
                formatted_sources = []
                for doc in sources:
                    meta = doc.metadata
                    source_path = meta.get("source", "Unknown")
                    lecture = os.path.basename(source_path).replace("_", " ")
                    total_pages = meta.get("total_pages", "N/A")
                    page = meta.get("page", "N/A")
                    formatted_sources.append(f"- source: {lecture}, total_pages: {total_pages}, page: {page + 1}")
                return "\n".join(list(set(formatted_sources)))  # remove duplicates

            source_info = format_sources(source_documents)
            result_to_show = result + "\n\nSources:\n" + source_info

            #response="Hi, I am MediBot!"
            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role':'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")
            
    st.markdown(
    """
    <style>
        .sticky-footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            padding: 10px 0;
            background-color: #0e1117;
            color: #ccc;
            font-size: 0.85em;
            z-index: 100;
            border-top: 1px solid #333;
        }

        .footer-content {
            text-align: center;
        }

        .footer-content a {
            color: #ccc;
            text-decoration: none;
            margin: 0 5px;
        }

        .footer-content a:hover {
            text-decoration: underline;
        }
    </style>

    <div class="sticky-footer">
        <div class="footer-content">
            <b>Hemant Rajput</b> |
            <a href="https://www.linkedin.com/in/hemant-rajput-2b8a8123a/" target="_blank">🔗 LinkedIn</a> |
            <a href="mailto:hemantrajput1201@gmail.com">📧 Email</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True)





if __name__ == "__main__":
    main()
