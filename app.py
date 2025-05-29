
import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load fine-tuned model and tokenizer
model_path = '/content/drive/MyDrive/Generative-Text-Model/fine_tuned_model'
try:
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("Loaded fine-tuned GPT-2 model and tokenizer.")
except Exception as e:
    st.error(f"Failed to load model or tokenizer: {str(e)}")
    raise e

# Set device and move model only if necessary
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    if model.device.type != device.type:
        model.to(device)
    logger.info(f"Model is on device: {model.device}")
except Exception as e:
    st.error(f"Failed to configure model device: {str(e)}")
    raise e
model.eval()  # Set model to evaluation mode

# Text generation function
def generate_text(prompt, max_length=100, temperature=0.7, top_k=50):
    if not prompt.strip():
        return "Error: Prompt cannot be empty."
    
    logger.info(f"Generating text for prompt: {prompt}")
    try:
        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
        
        # Move inputs to the same device as model
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        # Generate text
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id
        )
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info("Text generated successfully.")
        return generated_text.strip()
    except Exception as e:
        logger.error(f"Error during generation: {str(e)}")
        return f"Error during generation: {str(e)}"

# Streamlit UI
st.set_page_config(page_title="AI Text Generator", layout="centered")
st.markdown(
    """
    <style>
    .main {background-color: #ffffff; color: #000000;}
    .stButton>button {background-color: #FFC107; color: #000000; border-radius: 8px;}
    .stTextInput>div>input {border: 2px solid #FFC107; border-radius: 8px;}
    .generated-text {background-color: #F0F0F0; padding: 10px; border-radius: 8px;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("AI Text Generator with GPT-2")
st.write("Enter a prompt to generate a coherent paragraph using a fine-tuned GPT-2 model. (Note: Output may be limited due to small training dataset.)")

# Input components
prompt = st.text_input("Enter your prompt:", placeholder="e.g., Artificial intelligence is")
max_length = st.slider("Number of words to generate", 50, 200, 100)
generate_button = st.button("Generate Text")

# Generate and display text
if generate_button:
    if prompt:
        with st.spinner("Generating text..."):
            generated_text = generate_text(prompt, max_length=max_length, temperature=0.7, top_k=50)
            st.markdown(f"<div class='generated-text'>{generated_text}</div>", unsafe_allow_html=True)
    else:
        st.error("Please enter a prompt.")
