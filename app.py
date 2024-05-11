import streamlit as st
from pipeline.inference import generate_graphq_ir
import torch
import requests
from transformers import BartTokenizer, BartForConditionalGeneration
import time
import gdown
from transformers import BartConfig


def download_from_google_drive(gdrive_url, output_path):
    # Extract file id from the Google Drive link
    file_id = gdrive_url.split('/')[-2]
    download_url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(download_url, output_path, quiet=False)
    return output_path

@st.cache(allow_output_mutation=True)
def load_model_from_google_drive(gdrive_url, model_path='model.pth'):
    model = download_from_google_drive(gdrive_url, model_path)
    return model


config = BartConfig()


tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
model_path = "/Users/sarangsonar/Desktop/model.pth"
#downloaded_model = load_model_from_google_drive(model_url)
model = BartForConditionalGeneration(config)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()
device = torch.device("cpu")
model.to(device)


#streamlit UI

st.title("Text to Cypher")
st.write("This is a demo of the Text to Cypher pipeline. Please enter a sentence and click the button to generate a Cypher query.")
user_input = st.text_area("Enter your text:", "")


if st.button("Generate Cypher"):
    output = generate_graphq_ir(user_input, model, tokenizer,device)
    st.text_area("Cypher Query:", output)

