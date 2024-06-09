# Use the official Python image from the Docker Hub
FROM python:3.8-slim-buster

RUN apt update -y && apt install awscli -y


# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY . /app

# Install any dependencies specified in requirements.txt
RUN pip install -r requirements.txt


# Specify the command to run on container start
CMD ["python3", "app.py"]
