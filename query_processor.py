from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate


class QueryProcessor:
    def __init__(self, model_name="llama3.2"):
        """
        Initializes the QueryProcessor with a local Llama model.

        Args:
            model_name (str): Name of the local model to use.
        """
        self.llm = OllamaLLM(model=model_name)

    def generate_response(self, query, context):
        """
        Generates a response using the local Llama model.

        Args:
            query (str): The user's query.
            context (str): Context data to assist in generating the response.

        Returns:
            str: Generated response from the model.
        """
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
        # Combine query and context into the formatted prompt
        formatted_prompt = prompt_template.format(query=query, context=context)

        # Invoke the local Llama model
        response = self.llm.invoke(formatted_prompt)
        return response
