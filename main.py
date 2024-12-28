import streamlit as st
from transformers import AutoTokenizer, AutoModel
import torch
import os
from langchain_community.document_loaders import Docx2txtLoader
from chromadb import Client
import uuid

# Initialize Hugging Face model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Initialize ChromaDB client
client = Client()
collection = client.create_collection("resumes")


def get_embedding(text):
    """Generate an embedding for the given text using a pretrained model."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the mean of the last hidden state as the embedding
    return outputs.last_hidden_state.mean(dim=1).numpy()


# Streamlit UI
st.title("Resume Analysis Tool")

# Specify the folder containing resumes
resume_folder = "resource"

# Check if the folder exists and contains .docx files
if os.path.exists(resume_folder):
    resume_files = [f for f in os.listdir(resume_folder) if f.endswith(".docx")]

    if resume_files:
        for file_name in resume_files:
            file_path = os.path.join(resume_folder, file_name)
            # Load and parse resume using LangChain's Docx2txtLoader
            loader = Docx2txtLoader(file_path)
            documents = loader.load()

            # Assuming documents is a list of Document objects, extract text
            for doc in documents:
                resume_text = doc.page_content  # Extracting page content
                embedding = get_embedding(resume_text)
                # Store in ChromaDB with metadata (you can customize this)
                metadata = {"filename": file_name}
                collection.add(
                    documents=[resume_text],
                    embeddings=[embedding],
                    metadatas=[metadata],
                    ids=[str(uuid.uuid4())],
                )

        st.success(f"Processed {len(resume_files)} resumes successfully!")
    else:
        st.warning("No .docx files found in the resources folder.")
else:
    st.error(f"The folder '{resume_folder}' does not exist.")

# Input project requirements
project_requirements = st.text_area("Enter Project Requirements")

if st.button("Analyze"):
    if project_requirements:
        req_embedding = get_embedding(project_requirements)

        # Query candidates from ChromaDB based on similarity to project requirements
        results = collection.query(
            embedding=req_embedding, n_results=5
        )  # Adjust n_results as needed

        st.write("Recommended Candidates:")
        for candidate in results["documents"]:
            st.write(candidate)
    else:
        st.warning("Please enter project requirements before analyzing.")
