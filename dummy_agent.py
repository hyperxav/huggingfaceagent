from typing import List, Union, Dict
from transformers import Pipeline

class DummyAgent:
    def __init__(self, pipeline: Pipeline):
        """Initialize the DummyAgent with a pipeline.
        
        Args:
            pipeline (Pipeline): A Hugging Face pipeline for text generation
        """
        self.pipeline = pipeline
        self.conversation_history = []
        
    def run(self, user_input: str) -> str:
        """Process user input and return a response.
        
        Args:
            user_input (str): The user's input text
            
        Returns:
            str: The agent's response
        """
        # Add user input to conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Generate response using the pipeline
        response = self._generate_response(user_input)
        
        # Add agent response to conversation history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def _generate_response(self, user_input: str) -> str:
        """Generate a response using the pipeline.
        
        Args:
            user_input (str): The user's input text
            
        Returns:
            str: The generated response
        """
        # Format the input for the model
        prompt = self._format_conversation()
        
        # Generate response using the pipeline
        response = self.pipeline(prompt, max_length=100, num_return_sequences=1)[0]["generated_text"]
        
        # Clean up the response (remove the prompt from the generated text)
        response = response.replace(prompt, "").strip()
        
        return response
    
    def _format_conversation(self) -> str:
        """Format the conversation history into a single string.
        
        Returns:
            str: The formatted conversation history
        """
        formatted_conversation = ""
        for message in self.conversation_history:
            role = message["role"]
            content = message["content"]
            formatted_conversation += f"{role.capitalize()}: {content}\n"
        return formatted_conversation
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the conversation history.
        
        Returns:
            List[Dict[str, str]]: The conversation history
        """
        return self.conversation_history 