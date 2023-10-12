import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TextIteratorStreamer

model_id = 'Deci/DeciLM-6b-instruct'

model = AutoModelForCausalLM.from_pretrained(model_id,
                                             device_map="auto",
                                             trust_remote_code=True
                                             )

tokenizer = AutoTokenizer.from_pretrained(model_id)

tokenizer.pad_token = tokenizer.eos_token

SYSTEM_PROMPT_TEMPLATE = """
Write a dictionary of HTML attributes equivalent to the HTML element : " {instruction} "
"""

# Function to construct the prompt using the new system prompt template
def get_prompt_with_template(message: str) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(instruction=message)

# Function to handle the generation of the model's response using the constructed prompt
def generate_model_response(message: str) -> str:
    prompt = get_prompt_with_template(message)
    inputs = tokenizer(prompt, return_tensors='pt')
    if torch.cuda.is_available():  # Ensure input tensors are on the GPU if model is on GPU
        inputs = inputs.to('cuda')
    output = model.generate(**inputs,
                            max_new_tokens=3000,
                            num_beams=5,
                            no_repeat_ngram_size=4,
                            early_stopping=True
                            )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Function to extract the content after "### Response:"
def extract_response_content(full_response: str) -> str:
    response_start_index = full_response.find("### Response:")
    if response_start_index != -1:
        return full_response[response_start_index + len("### Response:"):].strip()
    else:
        return full_response
    

def get_response_with_template(message: str) -> str:
    full_response = generate_model_response(message)
    return extract_response_content(full_response)

def desc_to_dict(message : str) -> str:
    return get_response_with_template(message)