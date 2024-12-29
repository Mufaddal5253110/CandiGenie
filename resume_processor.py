import os
import uuid
import logging
import time
import pandas as pd
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb import PersistentClient
from transformers import AutoModel

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ResumeProcessor:
    def __init__(self, resume_folder="resource", collection_name="resumes"):
        self.resume_folder = resume_folder
        self.chroma_client = PersistentClient("vectorstore")
        self.collection = self.chroma_client.get_or_create_collection(collection_name)
        self.model = self._initialize_huggingface_model()

    def _initialize_huggingface_model(self):
        """Load the pre-trained Hugging Face model."""
        logging.info("Initializing Hugging Face model.")
        start_time = time.time()
        model = AutoModel.from_pretrained(
            "jinaai/jina-embeddings-v2-base-en", trust_remote_code=True
        )
        elapsed_time = time.time() - start_time
        logging.info("Model initialized in %.2f seconds.", elapsed_time)
        return model

    def get_embedding(self, text):
        """Generate an embedding for the given text using the pretrained model."""
        logging.info("Generating embedding for the provided text.")
        query_embeddings = self.model.encode(text).tolist()
        return query_embeddings

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

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=20
            )

            for doc in documents:
                resume_text = doc.page_content
                chunks = text_splitter.split_text(resume_text)

                for chunk in chunks:
                    embedding = self.get_embedding(chunk)
                    metadata = {"filename": file_name}

                    self.collection.add(
                        documents=[chunk],
                        embeddings=[embedding],
                        metadatas=[metadata],
                        ids=[str(uuid.uuid4())],
                    )

        total_elapsed_time = time.time() - total_start_time
        logging.info("Finished processing resumes in %.2f seconds.", total_elapsed_time)
        return total_elapsed_time
