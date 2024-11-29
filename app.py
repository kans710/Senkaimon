import streamlit as st
import os
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
import random
import warnings
from chat_template import css,bot_template,user_template
warnings.filterwarnings("ignore")

# Set environment variable for protobuf
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to create vector database from the PDF text
def create_vector_db(pdf_text):
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(pdf_text)

    # Wrap each chunk into a Document object
    documents = [Document(page_content=chunk) for chunk in chunks]
    st.write(f"Text split into {len(documents)} chunks")

    # Create vector database
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=OllamaEmbeddings(model="nomic-embed-text"),
        collection_name="local-rag"
    )
    st.write("Vector database created successfully")
    return vector_db

# Function to process a question using the RAG chain
def process_question(question, vector_db):
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
        Original question: {question}"""
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
        {"context": retriever, "question": RunnablePassthrough() }
        | prompt
        | llm
        | StrOutputParser()
    )

    # Get the response
    response = chain.invoke(question)
    return response

# Streamlit UI components
def main():
    st.set_page_config(page_title="Senkaimon", page_icon="🤖")

    # Custom CSS for the frosted, translucent effect on the main area and sidebar
    st.markdown(
        """
        <style>
        /* Import Poppins font from Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&display=swap');

        /* Apply Poppins font to the entire app */
        * {
            font-family: 'Poppins', sans-serif;
        }

        /* Style for the line with circles */
        .line-with-circles {
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 20px 0; /* Adjust spacing as needed */
        }

        .line-with-circles::before,
        .line-with-circles::after {
            content: '';
            flex: 1;
            height: 2px;
            background-color: #ffffff;
        }

        .circle {
            width: 8px; /* Circle diameter */
            height: 8px;
            margin: 0 10px; /* Spacing between the circles */
            background-color: #ffffff;
            border-radius: 50%;
        }

        /* Center-align the title and adjust font for clarity */
        .app-title {
            text-align: center;
            color: #ffffff;
            font-size: 2em;
            font-family: 'Poppins', sans-serif;
        }

        /* Apply a translucent gradient background to the main app area */
        .stApp {
            background: radial-gradient(ellipse at top, #081a32, #000000);
            padding: 20px;
            border-radius: 15px;
        }

        /* Style for the sidebar container */
        section[data-testid="stSidebar"] {
            background-color: radial-gradient(#e66465, #9198e5);
            border-radius: 15px;
            padding: 20px;
            background: rgba( 255, 255, 255, 0 );
            box-shadow: 0 8px 32px 0 rgba( 31, 38, 135, 0.37 );
            backdrop-filter: blur( 4px );
            -webkit-backdrop-filter: blur( 4px );
            border-radius: 10px;
            border: 1px solid rgba( 255, 255, 255, 0.18 );
        }

        /* Style the header and file uploader in the sidebar */
        section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] label {
            color: #ffffff;
            font-size: 1.1em;
            text-align: center;
        }

        /* Style the chat bubble (message) in cards */
        .chat-bubble {
            background-color: #0078d4;
            color: white;
            border-radius: 20px;
            padding: 10px 20px;
            margin: 10px;
            max-width: 70%;
            display: inline-block;
            position: relative;
        }

        .chat-bubble.user {
            background-color: #5a5a5a;
            float: right;
        }

        .chat-bubble.bot {
            background-color: #0084ff;
            float: left;
        }

        /* Style the message container */
        .message-container {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }

        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
    """
    <style>
    /* Fix the text input container at the bottom */
    .text-input-container {
        position: fixed;
        bottom: 20px;
        left: 20px;
        width: calc(100% - 40px); /* Adjust width based on margins */
        z-index: 1000; /* Ensure it stays above other elements */
    }

    /* Style the input box */
    .stTextInput > div {
        width: 100%; /* Full width */
    }

    /* Optional: Add padding at the bottom of the chat container to prevent overlap */
    .chat-container {
        padding-bottom: 80px; /* Adjust to match input height + margins */
    }
    </style>
    """,
    unsafe_allow_html=True
)


    # Sidebar content (file uploader)
    st.sidebar.header("Upload PDF")
    pdf_upload = st.sidebar.file_uploader("Upload a PDF document", type="pdf")

    # Main app content
    st.markdown("<h1 class='app-title'><u>The Senkaimon : Talk to your PDF '📚</u></h1>", unsafe_allow_html=True)
   


    if pdf_upload is not None:
        # Extract text from the uploaded PDF
        pdf_text = extract_text_from_pdf(pdf_upload)

        # Create vector database
        vector_db = create_vector_db(pdf_text)

        # Initialize a list to store the conversation
        if 'conversation' not in st.session_state:
            st.session_state.conversation = []

        # User chat interface
        st.markdown('<div class="text-input-container">', unsafe_allow_html=True)
        user_question = st.text_input("Ask a question about the PDF:")
        st.markdown('</div>', unsafe_allow_html=True)

        if user_question:
            with st.spinner("Processing your question..."):
                # Process the question and get the response
                response = process_question(user_question, vector_db)

                # Store conversation in session state
                st.session_state.conversation.append({"user": user_question, "bot": response})

        # Display conversation in cards
        if st.session_state.conversation:
            chat_container = st.container()
            with chat_container:
                for message in st.session_state.conversation:
                    # Display user message
                    st.markdown(f'<div class="chat-bubble user">{message["user"]}</div>', unsafe_allow_html=True)
                    # Display bot message
                    st.markdown(f'<div class="chat-bubble bot">{message["bot"]}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()