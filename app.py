import streamlit as st
from pipeline.inference import generate_graphq_ir
import torch
from transformers import BartTokenizer, BartForConditionalGeneration


tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
PATH="/Users/sarangsonar/Documents/GitHub/delfine-cypher_query/model/model.pth"
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
model.load_state_dict(torch.load(PATH))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#streamlit UI

st.title("Text to Cypher")
st.write("This is a demo of the Text to Cypher pipeline. Please enter a sentence and click the button to generate a Cypher query.")
user_input = st.text_area("Enter your text:", "")

if st.button("Generate Cypher"):
    output = generate_graphq_ir(user_input, model, tokenizer, device)
    st.text_area("Cypher Query:", output)

