FROM python:3.11-slim

WORKDIR /app

# Install git and git-lfs for model files
RUN apt-get update && apt-get install -y git git-lfs

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the port Chainlit will run on
EXPOSE 8000

# Command to run the application
CMD ["chainlit", "run", "app.py", "--host=0.0.0.0", "--port=8000"]