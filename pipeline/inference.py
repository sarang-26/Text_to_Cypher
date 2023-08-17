
def generate_graphq_ir(query, model, tokenizer, device):
    """
    Generate GraphQ IR sequence for a given natural language query.

    Args:
    - query (str): Natural language query.
    - model (BartForConditionalGeneration): Trained BART model.
    - tokenizer (BartTokenizer): BART tokenizer.
    - device (torch.device): Device (CPU or CUDA).

    Returns:
    - str: Generated GraphQ IR sequence.
    """
    # Ensure the model is in evaluation mode

    # Tokenize the input query
    inputs = tokenizer([query], return_tensors="pt", max_length=256, truncation=True).to(device)

    # Generate the output sequence
    output_ids = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=256, num_beams=5, temperature=0.7)

    # Decode the generated output
    decoded_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return decoded_output

