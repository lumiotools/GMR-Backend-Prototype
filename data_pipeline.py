import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import Pinecone
from langchain_openai.embeddings import OpenAIEmbeddings

# Load environment variables
load_dotenv()

def upload_text_to_pinecone(file_path, index_name='gmr-index'):
    # Load the text file
    loader = TextLoader(file_path)
    documents = loader.load()

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # Initialize embeddings and Pinecone vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = Pinecone.from_documents(
        texts, 
        embeddings, 
        index_name=index_name
    )

    print(f"Successfully uploaded {len(texts)} chunks to Pinecone index '{index_name}'")

# Example usage
if __name__ == "__main__":
    file_path = "data.txt"
    upload_text_to_pinecone(file_path)