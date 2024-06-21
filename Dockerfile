# Dockerfile
# Use Python 3.10 image from Docker Hub as the base image
FROM python:3.10-slim

# Update the package lists and install git in a single RUN command
RUN apt-get update && apt-get install -y git 

# Set the working directory inside the container to /app
WORKDIR /app

# Copy the requirements.txt file from the host to the current directory in the container
COPY requirements.txt ./

# Install the Python dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Ensure Streamlit is installed
RUN pip install streamlit

# Copy all files from the current directory on the host to the /app directory in the container
COPY ./app ./app
COPY ./.streamlit.example ./.streamlit.example

# Inform Docker that the container listens on port 8080 at runtime
ENV HOST=0.0.0.0
ENV LISTEN_PORT 8080
EXPOSE 8080

# Define environment variable
ENV PYTHONUNBUFFERED=1

# Set the default command to run the Streamlit application
ENTRYPOINT ["streamlit", "run"]
CMD ["app/streamlit.py",  "--server.port=8080", "--server.address=0.0.0.0"]
# , "--server.port=${PORT}"

