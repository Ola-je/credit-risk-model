# Use a lightweight Python base image
FROM python:3.10-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . /app

# Ensure MLflow's local storage path is recognized
# This assumes your mlruns directory is at the project root or relative path
# And that you've correctly registered a model named 'CreditRiskModel'
ENV MLFLOW_TRACKING_URI=sqlite:///mlruns/mlruns.db

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the application using Uvicorn
# The --host 0.0.0.0 makes it accessible from outside the container
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]