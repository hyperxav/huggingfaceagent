import os
from dummy_agent import DummyAgent

def main():
    # Get Hugging Face token from environment variable
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Warning: HF_TOKEN environment variable not set. You need to set it to use the model.")
        print("You can get a token from https://hf.co/settings/tokens")
        return

    # Create an instance of DummyAgent with GPT2
    agent = DummyAgent(model_id="gpt2", token=hf_token)
    
    # Example conversation
    print("Starting conversation with DummyAgent...")
    
    # Ask about weather
    response = agent.run("What's the weather in London?")
    print("Agent:", response)
    
    # Ask about another location
    response = agent.run("How about the weather in Paris?")
    print("Agent:", response)

if __name__ == "__main__":
    main() 