# Base image
FROM python:3.13

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the Streamlit port
EXPOSE 8501

# Healthcheck
HEALTHCHECK CMD streamlit hello

# Command to run the Streamlit app
# We will replace 'app/main.py' with the actual app entry point later
CMD ["streamlit", "run", "run.py"] 