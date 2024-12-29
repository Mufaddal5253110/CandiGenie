from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate


class QueryProcessor:
    def __init__(self, groq_api_key, model_name="llama-3.1-70b-versatile"):
        self.llm = ChatGroq(
            temperature=0, groq_api_key=groq_api_key, model_name=model_name
        )

    def generate_response(self, query, context):
        prompt_template = PromptTemplate.from_template(
            """
        ### QUERY:
        {query}

        ### CONTEXT DATA:
        {context}

        ### INSTRUCTION:
        Using the context data above, answer the query with as much precision as possible.
        Provide a clear and concise response.
        """
        )
        chain = prompt_template | self.llm
        return chain.invoke({"query": query, "context": context}).content
