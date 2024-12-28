import os
import uuid
import logging
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModel
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb import PersistentClient

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Initialize Hugging Face model and tokenizer
def initialize_huggingface_model():
    """Load the pre-trained Hugging Face model and tokenizer."""
    logging.info("Initializing Hugging Face model and tokenizer.")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    return tokenizer, model


def get_embedding(text, tokenizer, model):
    """Generate an embedding for the given text using a pretrained model.

    Args:
        text (str): Input text to embed.
        tokenizer: Hugging Face tokenizer.
        model: Hugging Face model.

    Returns:
        numpy.ndarray: Embedding vector for the input text.
    """
    logging.info("Generating embedding for the provided text.")
    embedding_model = AutoModel.from_pretrained(
        "jinaai/jina-embeddings-v2-base-en", trust_remote_code=True
    )
    user_query = text
    query_embeddings = embedding_model.encode(user_query).tolist()
    return query_embeddings

    # inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    # with torch.no_grad():
    #     outputs = model(**inputs)
    # return outputs.last_hidden_state.mean(dim=1).numpy()


# Initialize ChromaDB client and collection
def initialize_chromadb():
    """Initialize ChromaDB client and create a collection for resumes."""
    logging.info("Initializing ChromaDB client and collection.")
    client = PersistentClient("vectorstore")
    collection = client.get_or_create_collection("resumes")
    return collection


def process_resumes(resume_folder, tokenizer, model, collection):
    """Process resumes by loading, extracting text, generating embeddings, and storing in ChromaDB.

    Args:
        resume_folder (str): Path to the folder containing resume files.
        tokenizer: Hugging Face tokenizer.
        model: Hugging Face model.
        collection: ChromaDB collection object.
    """
    logging.info("Starting to process resumes from folder: %s", resume_folder)

    if not os.path.exists(resume_folder):
        st.error(f"The folder '{resume_folder}' does not exist.")
        logging.error("The folder '%s' does not exist.", resume_folder)
        return

    resume_files = [f for f in os.listdir(resume_folder) if f.endswith(".docx")]
    if not resume_files:
        st.warning("No .docx files found in the resources folder.")
        logging.warning("No .docx files found in the resources folder.")
        return

    with st.spinner("Processing resumes..."):
        for file_name in resume_files:
            logging.info("Processing file: %s", file_name)
            file_path = os.path.join(resume_folder, file_name)
            loader = Docx2txtLoader(file_path)
            documents = loader.load()

            # Split each document into chunks using LangChain's RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=300,  # Adjust chunk size as needed
                chunk_overlap=20,  # Adjust overlap as needed
            )

            for doc in documents:
                resume_text = doc.page_content  # Extracting page content
                # Create chunks from the resume text
                chunks = text_splitter.split_text(resume_text)

                for chunk in chunks:
                    # Generate embedding for each chunk
                    embedding = get_embedding(chunk, tokenizer, model)
                    metadata = {"filename": file_name}

                    collection.add(
                        documents=[chunk],
                        embeddings=[embedding],
                        metadatas=[metadata],
                        ids=[str(uuid.uuid4())],
                    )

    st.success(f"Processed {len(resume_files)} resumes successfully!")
    logging.info("Finished processing resumes.")


def analyze_requirements(project_requirements, collection, tokenizer, model):
    """Analyze project requirements and recommend candidates based on similarity.

    Args:
        project_requirements (str): Text describing project requirements.
        collection: ChromaDB collection object.
        tokenizer: Hugging Face tokenizer.
        model: Hugging Face model.
    """
    if not project_requirements:
        st.warning("Please enter project requirements before analyzing.")
        logging.warning("No project requirements entered.")
        return

    logging.info("Analyzing project requirements.")
    req_embedding = get_embedding(project_requirements, tokenizer, model)

    with st.spinner("Analyzing requirements and fetching results..."):
        results = collection.query(
            query_embeddings=req_embedding, n_results=5
        )  # Adjust n_results as needed

    st.write("Recommended Candidates:")
    for candidate in results.get("documents", []):
        st.write(candidate)
    logging.info("Analysis completed and results displayed.")


# Main Streamlit UI and application logic
def main():
    """Main function to run the Streamlit application."""
    st.title("CandiGenie")
    st.markdown(
        """
        ### Welcome to CandiGenie
        - Upload resumes in the 'resource' folder.
        - Enter project requirements to analyze and get candidate recommendations.
        """
    )

    # Initialize models and collections
    tokenizer, model = initialize_huggingface_model()
    collection = initialize_chromadb()

    # Specify the folder containing resumes
    resume_folder = "resource"

    # Process resumes
    process_resumes(resume_folder, tokenizer, model, collection)

    # Input project requirements
    project_requirements = st.text_area("Enter Project Requirements")

    if st.button("Analyze"):
        st.progress(0)
        analyze_requirements(project_requirements, collection, tokenizer, model)


if __name__ == "__main__":
    main()
