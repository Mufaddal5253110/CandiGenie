import logging
import time
import streamlit as st
from resume_processor import ResumeProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def analyze_requirements(project_requirements, resume_processor):
    """Analyze project requirements and recommend candidates."""
    if not project_requirements:
        st.warning("Please enter project requirements before analyzing.")
        logging.warning("No project requirements entered.")
        return 0

    logging.info("Analyzing project requirements.")
    start_time = time.time()
    req_embedding = resume_processor.get_embedding(project_requirements)

    with st.spinner("Analyzing requirements and fetching results..."):
        results = resume_processor.collection.query(
            query_embeddings=req_embedding, n_results=5
        )

    elapsed_time = time.time() - start_time
    st.write("Recommended Candidates:")
    for candidate in results.get("documents", []):
        st.write(candidate)
    st.success(f"Analysis completed in {elapsed_time:.2f} seconds!")
    logging.info("Analysis completed in %.2f seconds.", elapsed_time)
    return elapsed_time


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

    # Initialize ResumeProcessor
    resume_processor = ResumeProcessor(resume_folder="resource")

    # Load and process resumes if collection is empty
    if not resume_processor.collection.count():
        processing_time = resume_processor.load_and_process_resumes()
    else:
        processing_time = 0
        logging.info("Skipping resume processing as collection already contains data.")

    # Input project requirements
    project_requirements = st.text_area("Enter Project Requirements")

    if st.button("Analyze"):
        analysis_time = analyze_requirements(project_requirements, resume_processor)

        # Display timing information
        st.markdown("### Timing Summary")
        st.write(f"Resume Processing Time: {processing_time:.2f} seconds")
        st.write(f"Analysis Time: {analysis_time:.2f} seconds")


if __name__ == "__main__":
    main()
