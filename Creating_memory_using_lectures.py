from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Loading PDF files
path = "Big_Data_Lectures/"
def load_pdf_files(data):
    loader = DirectoryLoader(data, glob = '*.pdf', loader_cls = PyPDFLoader)
    documents = loader.load()
    
    return documents

documents = load_pdf_files(data=path)
# print('length of documents', len(documents))

# Creating chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_data)
    
    return text_chunks

text_chunks = create_chunks(extracted_data=documents)
# print("length of Chunks", len(text_chunks))

# Create Vector Embeddings
def create_embeddings():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    return embedding_model

embedding_model = create_embeddings()
    
# Store embeddings in vector database, FAISS
faiss_db_path = 'VectorEmbeddings/db_faiss'
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(faiss_db_path)
                                       
