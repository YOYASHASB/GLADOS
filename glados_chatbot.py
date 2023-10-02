from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the DialoGPT model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Set the conversation history
conversation_history = []

print("GLaDOS: Hello! How can I assist you today? (Type 'exit' to end)")

while True:
    user_input = input("User: ")

    if user_input.lower() == "exit":
        print("GLaDOS: Goodbye.")
        break

    # Add user input to the conversation history
    conversation_history.append(user_input)

    # Generate a response using DialoGPT
    input_text = " ".join(conversation_history)
    
    # Encode the input text to input_ids
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    
    # Generate a response using DialoGPT with attention_mask and pad_token_id
    response_ids = model.generate(input_ids, 
                                  max_length=150, 
                                  num_return_sequences=1, 
                                  no_repeat_ngram_size=2, 
                                  pad_token_id=tokenizer.eos_token_id,  # Set pad_token_id
                                  attention_mask=input_ids.new_ones(input_ids.shape))  # Set attention_mask
    
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True)

    # Display GLaDOS's response
    print(f"GLaDOS: {response}")

    # Add GLaDOS's response to the conversation history
    conversation_history.append(response)
