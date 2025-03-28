from typing import List, Dict, Any, Optional
from huggingface_hub import InferenceClient
import json

class DummyAgent:
    def __init__(self, model_id: str = "gpt2", token: Optional[str] = None):
        """Initialize the DummyAgent with a Hugging Face model.
        
        Args:
            model_id (str): The Hugging Face model ID to use
            token (Optional[str]): Hugging Face API token
        """
        self.client = InferenceClient(model_id, token=token)
        self.conversation_history = []
        self.available_tools = {
            "get_weather": {
                "description": "Get the current weather in a given location",
                "args": {"location": {"type": "string"}}
            }
        }
        
    def _get_system_prompt(self) -> str:
        """Get the system prompt that describes available tools."""
        return """You are a helpful assistant that can check the weather. Use the get_weather tool by responding with:
Action:
```
{"action": "get_weather", "action_input": {"location": "CITY_NAME"}}
```
Then wait for the Observation and provide a Final Answer."""

    def _format_prompt(self, user_input: str) -> str:
        """Format the conversation into a prompt the model expects."""
        return f"{self._get_system_prompt()}\n\nUser: {user_input}\nAssistant: Let me check that for you.\n"
        
    def _get_weather(self, location: str) -> str:
        """Dummy weather function.
        
        Args:
            location (str): The location to get weather for
            
        Returns:
            str: A dummy weather response
        """
        return f"the weather in {location} is sunny with low temperatures.\n"
        
    def _execute_tool(self, action: str, action_input: Dict[str, Any]) -> str:
        """Execute a tool based on the action and input.
        
        Args:
            action (str): The name of the tool to execute
            action_input (Dict[str, Any]): The input parameters for the tool
            
        Returns:
            str: The tool's response
        """
        if action == "get_weather":
            return self._get_weather(action_input["location"])
        raise ValueError(f"Unknown tool: {action}")
        
    def run(self, user_input: str) -> str:
        """Process user input and return a response.
        
        Args:
            user_input (str): The user's input text
            
        Returns:
            str: The agent's response
        """
        # Format the initial prompt
        prompt = self._format_prompt(user_input)
        
        while True:
            # Generate the next part of the conversation
            response = self.client.text_generation(
                prompt,
                max_new_tokens=50,
                stop=["Observation:", "User:"]
            )
            
            # Add the response to the prompt
            prompt += response
            
            # Check if we've reached a final answer
            if "Final Answer:" in response:
                final_answer = response.split("Final Answer:")[-1].strip()
                return final_answer
                
            # Try to extract and execute tool calls
            try:
                # Find the JSON blob between ```
                action_start = response.find("```") + 3
                action_end = response.find("```", action_start)
                if action_start > 2 and action_end > action_start:
                    action_blob = response[action_start:action_end].strip()
                    action_data = json.loads(action_blob)
                    
                    # Execute the tool
                    result = self._execute_tool(
                        action_data["action"],
                        action_data["action_input"]
                    )
                    
                    # Add the observation to the prompt
                    prompt += f"\nObservation: {result}\n"
                else:
                    # No tool call found, just continue
                    continue
                    
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                # If there's an error parsing or executing the tool, return an error message
                return f"Error: {str(e)}" 