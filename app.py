from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.chat_models import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema import Document
import os
from IPython.display import display, Markdown
import warnings

warnings.filterwarnings("ignore")

# Set environment variable for protobuf
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Extract text from the PDF using PyPDF2
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

local_path = "AI_Roadmap.pdf"
if local_path:
    pdf_text = extract_text_from_pdf(local_path)
    print(f"PDF loaded successfully: {local_path}")
else:
    print("Upload a PDF file")

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_text(pdf_text)

# Wrap each chunk into a Document object
documents = [Document(page_content=chunk) for chunk in chunks]
print(f"Text split into {len(documents)} chunks")


vector_db = Chroma.from_documents(
    documents=documents,
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    collection_name="local-rag"
)
print("Vector database created successfully")

# Set up LLM and retrieval
local_model = "llama3.2"
llm = ChatOllama(model=local_model)

# Query prompt template
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate 2
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

# Set up retriever
retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(),
    llm,
    prompt=QUERY_PROMPT
)

# RAG prompt template
template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# Create chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def chat_with_pdf(question):
    """
    Chat with the PDF using the RAG chain.
    """
    return display(Markdown(chain.invoke(question)))

# Example 1
chat_with_pdf("What is this document about?")

# Optional: Clean up when done
# vector_db.delete_collection()
# print("Vector database deleted successfully")
