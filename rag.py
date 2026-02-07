# !pip install langchain langchain-google-genai langchain-classic langchain-community faiss-cpu langchain-text-splitters
# !pip install pypdf
# !pip install sentence-transformers


from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough

## document loading
loader = PyPDFLoader("Blue_IT_Company_Terms_and_Conditions.pdf")
document = loader.load()
print("Length of the documents:",len(document))

print("\ndocument:\n", document[0].page_content[:])

## Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 60
)

chunk = splitter.split_documents(document)
print("\nlenght of chunk:", len(chunk))
print("\nChunk\n:",chunk[0].page_content[:])

## Embedding
embedding = HuggingFaceEmbeddings(
    model_name = "all-MiniLM-L6-V2"
)

## Vector Database
vectorstore = FAISS.from_documents(
    chunk,
    embedding
)

retriever = vectorstore.as_retriever(
    search_kwargs = {"k":2}
)

res = retriever.invoke("What is the main core of the Blue It company?")
for r in res:
    print(r.page_content)

def format_doc(docs):
    return "\n\n".join(doc.page_content for doc in docs)


## Memory
memory = ConversationBufferMemory(
    memory_key = "chat_history"
)
print("\n Initial Length of memory: ", len(memory.chat_memory.messages))


## Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpfull AI assistant. Use the Context to answer the question"),
    ("system", "chat_history: {chat_history}"),
    ("human", "context: {context}"),
    ("human", "question: {question}")
])

## API Key
import os
from API_key import GEMINI_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY


#LLM
llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    temperature = 0.2
)
llm

#RAG Chain
rag_chain = (
    {
        "context": retriever | format_doc,
        "question": RunnablePassthrough(),
        "chat_history": lambda _: memory.chat_memory.messages
    }
    | prompt
    | llm
)


query = input("Enter your query: ")
response= rag_chain.invoke(query)
print(f"Your Query: \n {query} \n\n Answer: \n {response.content}")

# Saving Memory
memory.save_context(
    {"input":query},
    {"output":response.content}
)


