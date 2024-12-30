import logging
import time
from query_processor import QueryProcessor
import streamlit as st
from resume_processor import ResumeProcessor
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

vector_results_count = 5

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def analyze_requirements(query, resume_processor, query_processor):
    """
    Analyze project requirements, recommend candidates, and generate a meaningful answer.

    Args:
        query (str): The user's query or project requirements.
        resume_processor (ResumeProcessor): Instance for handling resumes and embeddings.
        query_processor (QueryProcessor): Instance for interacting with the LLM.

    Returns:
        dict: Timing details for the analysis process.
    """
    if not query:
        st.warning("Please enter project requirements before analyzing.")
        logging.warning("No project requirements entered.")
        return {}

    logging.info("Starting analysis for query: %s", query)

    # Measure time for vector search
    start_time = time.time()
    req_embedding = resume_processor.get_embedding(query)

    with st.spinner("Fetching relevant documents from the vector database..."):
        vector_start_time = time.time()
        results = resume_processor.collection.query(
            query_embeddings=req_embedding, n_results=vector_results_count
        )
        vector_elapsed_time = time.time() - vector_start_time

    logging.info("Vector search completed in %.2f seconds.", vector_elapsed_time)

    # Initialize processed chunks and timing
    processed_chunks = []
    llm_elapsed_times = []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    # Process each document with LLM
    for doc, meta in zip(documents, metadatas):
        context = f"Filename: {meta.get('filename', 'N/A')}\n\n{doc}"
        llm_start_time = time.time()
        processed_response = query_processor.generate_individual_response(
            query, context
        )
        llm_elapsed_time = time.time() - llm_start_time
        llm_elapsed_times.append(llm_elapsed_time)
        processed_chunks.append(processed_response)

        logging.info("Processed document with LLM in %.2f seconds.", llm_elapsed_time)

    # Final LLM call
    final_context = "\n".join(processed_chunks)
    final_llm_start_time = time.time()
    final_response = query_processor.generate_final_response(query, final_context)
    final_llm_elapsed_time = time.time() - final_llm_start_time

    logging.info("Final LLM call completed in %.2f seconds.", final_llm_elapsed_time)

    # Total elapsed time
    total_elapsed_time = time.time() - start_time
    logging.info("Total analysis time: %.2f seconds.", total_elapsed_time)

    # Display results
    st.success(f"Analysis completed in {total_elapsed_time:.2f} seconds!")

    return {
        "vector_search": vector_elapsed_time,
        "llm_calls": llm_elapsed_times,
        "final_llm_call": final_llm_elapsed_time,
        "final_response": final_response,
        "total_analysis": total_elapsed_time,
    }


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

    # Initialize resume processor and query processor
    resume_processor = ResumeProcessor(resume_folder="resource")
    query_processor = QueryProcessor(groq_api_key=os.getenv("GROQ_API_KEY"))

    # Load and process resumes if collection is empty
    if not resume_processor.collection.count():
        processing_time = resume_processor.load_and_process_resumes()
    else:
        processing_time = 0
        logging.info("Skipping resume processing as collection already contains data.")

    # Input project requirements
    query = st.text_area("Enter Project Requirements")

    if st.button("Analyze"):
        analysis_results = analyze_requirements(
            query, resume_processor, query_processor
        )

        # Display question and answer together
        st.markdown("### Answer:")
        st.write(analysis_results["final_response"])

        # Display timing information in tabular form
        timing_data = {
            "Step": [
                "Resume Processing Time",
                "Vector Search Time",
                "Total LLM Calls",
                "LLM Calls Time",
                "Final LLM Call Time",
                "Total Analysis Time",
                "Total Time",
            ],
            "Data": [
                processing_time,
                f'{analysis_results["vector_search"]} seconds',
                f"{vector_results_count + 1} count",
                f'{sum(analysis_results["llm_calls"])} seconds',
                f'{analysis_results["final_llm_call"]} seconds',
                f'{analysis_results["total_analysis"]} seconds',
                f'{sum([processing_time, analysis_results["vector_search"], sum(analysis_results["llm_calls"]), analysis_results["final_llm_call"], analysis_results["total_analysis"]])} seconds',
            ],
        }

        timing_df = pd.DataFrame(timing_data)

        st.markdown("### Summary")
        st.table(timing_df)


if __name__ == "__main__":
    main()
