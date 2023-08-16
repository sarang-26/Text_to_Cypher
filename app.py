import streamlit as st
from pipeline.inference import generate_graphq_ir
import torch
from transformers import BartTokenizer, BartForConditionalGeneration




@st.cache(allow_output_mutation=True)
def load_model_from_url(url, model_path='model.pth'):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(model_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
    model.load_state_dict(torch.load(PATH))
    model.eval()
    return model

# Load the model
model_url = "https://drive.google.com/file/d/1-DvdMr0vIJKKB3Efqz-zrImQcpOuc5Nq/view?usp=drive_link"
model = load_model_from_url(model_url)
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


#streamlit UI

st.title("Text to Cypher")
st.write("This is a demo of the Text to Cypher pipeline. Please enter a sentence and click the button to generate a Cypher query.")
user_input = st.text_area("Enter your text:", "")


if st.button("Generate Cypher"):
    output = generate_graphq_ir(user_input, model, tokenizer, device)
    st.text_area("Cypher Query:", output)

