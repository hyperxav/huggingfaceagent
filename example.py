from transformers import pipeline
from dummy_agent import DummyAgent

def main():
    # Initialize a text generation pipeline
    text_pipeline = pipeline("text-generation", model="gpt2")
    
    # Create an instance of DummyAgent
    agent = DummyAgent(text_pipeline)
    
    # Example conversation
    print("Starting conversation with DummyAgent...")
    
    # First user input
    response = agent.run("Hello! How are you?")
    print("Agent:", response)
    
    # Second user input
    response = agent.run("What can you help me with?")
    print("Agent:", response)
    
    # Print conversation history
    print("\nConversation History:")
    for message in agent.get_conversation_history():
        print(f"{message['role'].capitalize()}: {message['content']}")

if __name__ == "__main__":
    main() 