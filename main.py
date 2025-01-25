import os
from fastapi import FastAPI
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_pinecone import PineconeVectorStore
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()



class ChatModel(BaseModel):
    query: str


# Initialize FastAPI app
app = FastAPI()

# Initialize models and vector store
model = ChatOpenAI(model="gpt-4o-mini")
vectorstore = PineconeVectorStore(
    index_name="gmr-index",
    embedding=OpenAIEmbeddings()
)

# Prompt template
template = """
You are an AI avatar assistant stationed at Hyderabad Airport. Your primary role is to assist travelers by providing accurate and helpful answers to their queries. Always be clear, short, concise, and contextually relevant, ensuring the information is tailored to the user's needs.

Context: {context}
User Question: {question}
"""

def get_top_3_context(query):
    retriever = vectorstore.similarity_search_with_score(query=query, k=3)
    text = "\n".join([str({doc[0].page_content}) for i, doc in enumerate(retriever)])
    return text


prompt = ChatPromptTemplate.from_template(template)
chain = (
    {"context": get_top_3_context, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

@app.post("/chat/")
def chat(query: ChatModel):
    return chain.invoke(query.query)


# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=(os.getenv("ENV") == "dev"))
