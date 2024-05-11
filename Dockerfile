FROM huggingface/transformers-pytorch-gpu

# Set the working directory in docker
WORKDIR /app

# Copy Python dependencies
COPY requirements.txt .

# Install any additional Python dependencies that aren't in the base image
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the content of the local src directory to the working directory
COPY src/ .

# Expose port for Streamlit
EXPOSE 8501

# Specify the command to run on container start
CMD ["streamlit", "run", "./app.py"]
