from pathlib import Path
import sys

import tiktoken
import torch
import gradio as gr

from previous_chapters import (
    generate,
    GPTModel,
    text_to_token_ids,
    token_ids_to_text,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_and_tokenizer():
    """
    Code to load a GPT-2 model with finetuned weights generated in chapter 7.
    This requires that you run the code in chapter 7 first, which generates the necessary gpt2-medium355M-sft.pth file.
    """
    
    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Shortened context length (orig: 1024)
        "emb_dim": 768,         # Embedding dimension
        "n_heads": 12,           # Number of attention heads
        "n_layers": 12,          # Number of layers
        "drop_rate": 0.0,        # Dropout rate
        "qkv_bias": True         # Query-key-value bias
    }

    tokenizer = tiktoken.get_encoding("gpt2")

    model_path = Path(".") / "gpt2-small124M-sft.pth"
    if not model_path.exists():
        print(
            f"Could not find the {model_path} file. Please run the chapter 7 code "
            " (ch07.ipynb) to generate the gpt2-small124M-sft.pt file."
        )
        sys.exit()

    checkpoint = torch.load(model_path, weights_only=True)
    model = GPTModel(GPT_CONFIG_124M)
    model.load_state_dict(checkpoint)
    model.to(device)

    return tokenizer, model, GPT_CONFIG_124M


def extract_response(response_text, input_text):
    return response_text[len(input_text):].replace("### Response:", "").strip()


# Load the model and tokenizer once when the app starts
tokenizer, model, model_config = get_model_and_tokenizer()

def predict(message, history):
    """
    Generate response from the model based on user input.
    """
    torch.manual_seed(123)

    prompt = f"""Below is an instruction that describes a task. Write a response
    that appropriately completes the request.

    ### Instruction:
    {message}
    """

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(prompt, tokenizer).to(device),
        max_new_tokens=35,
        context_size=model_config["context_length"],
        eos_id=50256
    )

    text = token_ids_to_text(token_ids, tokenizer)
    response = extract_response(text, prompt)
    
    return response

# Create Gradio Interface
demo = gr.ChatInterface(
    fn=predict,
    title="Instructions based Large Language Model",
    description="Chat with a self made Large Language Model that can generate responses based upon prompts.",
    examples=["Tell me a short story", "Explain quantum physics", "What is machine learning?"],
    theme="default"
)

# Launch the app
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=True)