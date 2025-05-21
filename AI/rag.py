from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import pickle
import os

# Set file paths
pdfs_folder = r"C:\Users\daksh\ragg\pdf_files"  # Folder containing PDFs
pkl_folder = r"C:\Users\daksh\ragg\pkl_files"  # Folder to store .pkl files
pdf_file = os.path.join(pdfs_folder, "subhas.pdf")
faiss_index_path = os.path.join(pkl_folder, "faiss_index.pkl")

# Create pkl_folder if it doesn't exist
if not os.path.exists(pkl_folder):
    os.makedirs(pkl_folder)
    print(f"Created directory: {pkl_folder}")

# Function to save FAISS index as a pickle file
def save_faiss_to_pickle(faiss_index, file_path=faiss_index_path):
    with open(file_path, "wb") as f:
        pickle.dump(faiss_index, f)
    print(f"FAISS index saved as a pickle file: {file_path}")

# Function to load FAISS index from a pickle file
def load_faiss_from_pickle(file_path=faiss_index_path):
    with open(file_path, "rb") as f:
        faiss_index = pickle.load(f)
    print("FAISS index loaded successfully from pickle file.")
    return faiss_index

# Function to create FAISS index and save as a pickle file
def create_faiss_index_and_save(documents, file_path=faiss_index_path):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    db = FAISS.from_documents(documents, embeddings)
    save_faiss_to_pickle(db, file_path)
    print("FAISS index created and saved successfully!")

def initialize_system():
    """
    Initialize the RAG system by either loading an existing index or creating a new one
    """
    # Check if the index file exists
    if os.path.exists(faiss_index_path):
        print("Loading existing vector database...")
        # Load the saved index
        db = load_faiss_from_pickle(faiss_index_path)
    else:
        print("Creating new vector database from PDF...")
        # Check if PDF exists
        if not os.path.exists(pdf_file):
            print(f"Error: PDF file not found at {pdf_file}")
            exit(1)
            
        # Load documents from PDF file
        loader = PyPDFLoader(pdf_file)
        docs = loader.load()
        
        # Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(docs)
        
        # Create embeddings and vector database
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        db = FAISS.from_documents(documents, embeddings)
        
        # Save the vector database for future use
        save_faiss_to_pickle(db, faiss_index_path)
    
    # Initialize Ollama LLM
    try:
        llm = Ollama(model="llama2", base_url="http://localhost:11434")
        print("LLM initialized successfully!")
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        print("Make sure the Ollama server is running (use 'ollama serve' command)")
        exit(1)
    
    # Create a conversational prompt template that includes chat history
    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based **only** on the provided context.  
    If the context **does not contain relevant information**, respond with **"I don't have enough information to answer this."**  

    Provide a **concise and factual** response, avoiding unnecessary details.

    <chat_history>
    {history}
    </chat_history>

    <context>  
    {context}  
    </context>  

    Question: {input}  
    """)
    
    # Create document chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Initialize retriever
    retriever = db.as_retriever()
    
    # Create retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain

# Store conversation history
chat_history = []

def chat_with_ai(retrieval_chain, user_input):
    """
    Function to handle chat interactions with the AI model.
    It keeps track of the chat history and retrieves relevant context.
    """
    global chat_history  # Keep track of previous messages
    
    # Format history as a single string (keeping last 10 exchanges)
    history_text = "\n".join(chat_history[-10:])
    
    try:
        # Query the retrieval chain
        response = retrieval_chain.invoke({
            "history": history_text,
            "input": user_input
        })
        
        # Get the assistant's response
        ai_response = response['answer']
        
        # Update chat history
        chat_history.append(f"User: {user_input}")
        chat_history.append(f"Assistant: {ai_response}")
        
        return ai_response
    except Exception as e:
        return f"Error: {str(e)}. Make sure the Ollama server is running."

def main():
    """
    Main function to run the interactive chat system
    """
    print("Initializing Subhas Chandra Bose Chatbot...")
    print("This chatbot can answer questions about Subhas Chandra Bose based on provided documents.")
    
    # Initialize the system
    retrieval_chain = initialize_system()
    
    print("\n--------------------------------------------------")
    print("Chatbot is ready! You can now ask questions about Subhas Chandra Bose.")
    print("Type 'exit', 'quit', or 'stop' to end the conversation.")
    print("--------------------------------------------------")
    
    # Run the chat system interactively
    while True:
        user_input = input("\nYou: ")  # Get user input
        
        if user_input.lower() in ["exit", "quit", "stop"]:
            print("Chat session ended.")
            break
        
        response = chat_with_ai(retrieval_chain, user_input)
        print(f"Assistant: {response}")

if __name__ == "__main__":
    main()






# import os
# import pickle
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings

# # Function to save FAISS index as a pickle file
# def save_faiss_to_pickle(faiss_index, file_path):
#     with open(file_path, "wb") as f:
#         pickle.dump(faiss_index, f)
#     print(f"FAISS index saved as a pickle file: {file_path}")

# # Function to create FAISS index and save as a pickle file
# def create_faiss_index_and_save(documents, model_name, file_path):
#     if not documents:
#         print(f"No documents found for {file_path}. Skipping.")
#         return
    
#     embeddings = HuggingFaceEmbeddings(model_name=model_name, encode_kwargs={"normalize_embeddings": True})
#     db = FAISS.from_documents(documents, embeddings)
#     save_faiss_to_pickle(db, file_path)
#     print(f"FAISS index created and saved for {file_path}!")

# # Function to process each PDF separately
# def process_pdfs_separately(pdf_folder, pkl_folder, model_name):
#     if not os.path.exists(pkl_folder):
#         os.makedirs(pkl_folder)
    
#     for file in os.listdir(pdf_folder):
#         if file.endswith(".pdf"):
#             pdf_path = os.path.join(pdf_folder, file)
#             loader = PyPDFLoader(pdf_path)
#             try:
#                 documents = loader.load()
#                 pkl_file_path = os.path.join(pkl_folder, f"{os.path.splitext(file)[0]}.pkl")
#                 create_faiss_index_and_save(documents, model_name, pkl_file_path)
#             except Exception as e:
#                 print(f"Error processing {file}: {e}")

# # Example usage
# pdfs = r"D:\Code\RAG\Lang_rag\pdfs"  # Folder containing PDFs
# pkl_folder = r"D:\Code\RAG\Lang_rag\pkl_files"  # Folder to store .pkl files
# model_name = "sentence-transformers/all-mpnet-base-v2"
# process_pdfs_separately(pdfs, pkl_folder, model_name)
