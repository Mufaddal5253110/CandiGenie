import os
import uuid
import logging
import time
import torch
import streamlit as st
from transformers import AutoModel
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb import PersistentClient

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Initialize Hugging Face model
def initialize_huggingface_model():
    """Load the pre-trained Hugging Face model."""
    logging.info("Initializing Hugging Face model.")
    start_time = time.time()
    model = AutoModel.from_pretrained(
        "jinaai/jina-embeddings-v2-base-en", trust_remote_code=True
    )
    elapsed_time = time.time() - start_time
    logging.info("Model initialized in %.2f seconds.", elapsed_time)
    return model, elapsed_time


def get_embedding(text, model):
    """Generate an embedding for the given text using a pretrained model.

    Args:
        text (str): Input text to embed.
        model: Hugging Face model.

    Returns:
        numpy.ndarray: Embedding vector for the input text.
    """
    logging.info("Generating embedding for the provided text.")

    user_query = text
    query_embeddings = model.encode(user_query).tolist()
    return query_embeddings


# Initialize ChromaDB client and collection
def initialize_chromadb():
    """Initialize ChromaDB client and create a collection for resumes."""
    logging.info("Initializing ChromaDB client and collection.")
    start_time = time.time()
    client = PersistentClient("vectorstore")
    collection = client.get_or_create_collection("resumes")
    elapsed_time = time.time() - start_time
    logging.info("ChromaDB initialized in %.2f seconds.", elapsed_time)
    return collection, elapsed_time


def process_resumes(resume_folder, model, collection):
    """Process resumes by loading, extracting text, generating embeddings, and storing in ChromaDB.

    Args:
        resume_folder (str): Path to the folder containing resume files.
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

    total_start_time = time.time()
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
                    embedding = get_embedding(chunk, model)
                    metadata = {"filename": file_name}

                    collection.add(
                        documents=[chunk],
                        embeddings=[embedding],
                        metadatas=[metadata],
                        ids=[str(uuid.uuid4())],
                    )

    total_elapsed_time = time.time() - total_start_time
    st.success(
        f"Processed {len(resume_files)} resumes successfully in {total_elapsed_time:.2f} seconds!"
    )
    logging.info("Finished processing resumes in %.2f seconds.", total_elapsed_time)
    return total_elapsed_time


def analyze_requirements(project_requirements, collection, model):
    """Analyze project requirements and recommend candidates based on similarity.

    Args:
        project_requirements (str): Text describing project requirements.
        collection: ChromaDB collection object.
        model: Hugging Face model.
    """
    if not project_requirements:
        st.warning("Please enter project requirements before analyzing.")
        logging.warning("No project requirements entered.")
        return

    logging.info("Analyzing project requirements.")
    start_time = time.time()
    req_embedding = get_embedding(project_requirements, model)

    with st.spinner("Analyzing requirements and fetching results..."):
        results = collection.query(
            query_embeddings=req_embedding, n_results=5
        )  # Adjust n_results as needed

    elapsed_time = time.time() - start_time
    st.write("Recommended Candidates:")
    for candidate in results.get("documents", []):
        st.write(candidate)
    st.success(f"Analysis completed in {elapsed_time:.2f} seconds!")
    logging.info("Analysis completed in %.2f seconds.", elapsed_time)
    return elapsed_time


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
    model, model_time = initialize_huggingface_model()
    collection, chromadb_time = initialize_chromadb()

    # Specify the folder containing resumes
    resume_folder = "resource"

    # Process resumes
    resume_processing_time = process_resumes(resume_folder, model, collection)

    # Input project requirements
    project_requirements = st.text_area("Enter Project Requirements")

    if st.button("Analyze"):
        st.progress(0)
        analysis_time = analyze_requirements(project_requirements, collection, model)

        # Display timing information
        st.markdown("### Timing Summary")
        st.write(f"Model Initialization: {model_time:.2f} seconds")
        st.write(f"ChromaDB Initialization: {chromadb_time:.2f} seconds")
        st.write(f"Resume Processing: {resume_processing_time:.2f} seconds")
        st.write(f"Analysis Time: {analysis_time:.2f} seconds")


if __name__ == "__main__":
    main()
