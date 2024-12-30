from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate


class QueryProcessor:
    def __init__(self, groq_api_key, model_name="llama-3.3-70b-versatile"):
        self.llm = ChatGroq(
            temperature=0, groq_api_key=groq_api_key, model_name=model_name
        )

    def generate_individual_response(self, query, context):
        prompt_template = PromptTemplate.from_template(
            """
            ### INSTRUCTION:
            You are a highly skilled assistant specialized in analyzing and summarizing text. Analyze the following context and extract key details or insights that directly relate to the given query.

            ### QUERY:
            {query}

            ### CONTEXT:
            {context}

            ### OUTPUT REQUIREMENTS:
            1. Extract only the information relevant to the query.
            2. Provide concise and clear insights or details.
            3. Maintain a professional tone and avoid unnecessary details.

            ### RESPONSE:
            """
        )
        chain = prompt_template | self.llm
        return chain.invoke({"query": query, "context": context}).content

    def generate_final_response(self, query, context):
        prompt_template = PromptTemplate.from_template(
            """
            ### INSTRUCTION:
            You are an expert in aggregating and synthesizing information. Using the provided context from multiple sources, generate a comprehensive and precise response to the given query.

            ### QUERY:
            {query}

            ### CONTEXT FROM PREVIOUS RESPONSES:
            {context}

            ### OUTPUT REQUIREMENTS:
            1. Integrate relevant information from the context into a cohesive and complete response.
            2. Avoid repetition or contradictions; ensure logical flow and clarity.
            3. Provide the final answer in a concise and professional tone.

            ### FINAL RESPONSE:
            """
        )
        chain = prompt_template | self.llm
        return chain.invoke({"query": query, "context": context}).content
