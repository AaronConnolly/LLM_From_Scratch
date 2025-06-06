import torch
import tiktoken
from LLMScratch import GPT, GPTConfig  # Import your GPT model and configuration

# Set device
device = 'cpu'
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"Using device: {device}")

# Load the model
checkpoint_path = "model_checkpoint_step_100.pt"  # Update this to your checkpoint file
config = GPTConfig(vocab_size=50304, block_size=1024, n_layer=4, n_head=8, n_embd=256)  # Match your training config
model = GPT(config)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)
model.eval()
print(f"Loaded model from {checkpoint_path}")

# Initialize tokenizer
enc = tiktoken.get_encoding("gpt2")

# Function to generate text
def generate_text(prompt, max_length=50, num_return_sequences=1):
    # Encode the input prompt
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)  # Shape: (1, T)
    
    # Generate text
    with torch.no_grad():
        for _ in range(max_length):
            logits, _ = model(tokens)
            logits = logits[:, -1, :]  # Get logits for the last token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # Sample the next token
            tokens = torch.cat((tokens, next_token), dim=1)  # Append the new token
            
            # Stop if the end-of-text token is generated
            if next_token.item() == enc.eot_token:
                break
    
    # Decode the generated tokens
    generated_text = enc.decode(tokens[0].tolist())
    return generated_text

# Test the model
if __name__ == "__main__":
    prompt = input("Enter a prompt: ")
    generated_text = generate_text(prompt, max_length=50)
    print("\nGenerated Text:")
    print(generated_text)