import os
import uuid
import logging
import time
import pandas as pd
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from chromadb import PersistentClient

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ResumeProcessor:
    def __init__(self, resume_folder="resource", collection_name="resumes"):
        self.resume_folder = resume_folder
        self.chroma_client = PersistentClient("vectorstore")
        self.collection = self.chroma_client.get_or_create_collection(collection_name)
        self.embedding_model = self._initialize_huggingface_model()

    def _initialize_huggingface_model(self):
        """Load the pre-trained Hugging Face model."""
        logging.info("Initializing Hugging Face model.")
        start_time = time.time()
        model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        elapsed_time = time.time() - start_time
        logging.info("Model initialized in %.2f seconds.", elapsed_time)
        return model

    def get_embedding(self, text):
        """Generate an embedding for the given text using the HuggingFaceEmbeddings model."""
        logging.info("Generating embedding for the provided text.")
        return self.embedding_model.embed_query(text)

    def load_and_process_resumes(self):
        """Load resumes, generate embeddings, and store them in ChromaDB."""
        if not os.path.exists(self.resume_folder):
            logging.error("The folder '%s' does not exist.", self.resume_folder)
            return

        resume_files = [
            f for f in os.listdir(self.resume_folder) if f.endswith(".docx")
        ]
        if not resume_files:
            logging.warning("No .docx files found in the resources folder.")
            return

        logging.info("Starting to process resumes from folder: %s", self.resume_folder)
        total_start_time = time.time()
        for file_name in resume_files:
            logging.info("Processing file: %s", file_name)
            file_path = os.path.join(self.resume_folder, file_name)
            loader = Docx2txtLoader(file_path)
            documents = loader.load()

            for doc in documents:
                resume_text = doc.page_content

                embedding = self.get_embedding(resume_text)
                metadata = {"filename": file_name}

                self.collection.add(
                    documents=[resume_text],
                    embeddings=[embedding],
                    metadatas=[metadata],
                    ids=[str(uuid.uuid4())],
                )

        total_elapsed_time = time.time() - total_start_time
        logging.info("Finished processing resumes in %.2f seconds.", total_elapsed_time)
        return total_elapsed_time
