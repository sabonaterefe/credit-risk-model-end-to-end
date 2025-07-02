# Use a lightweight Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Ensure the start.sh script uses Unix line endings and is executable
RUN sed -i 's/\r$//' start.sh && chmod +x start.sh

# Expose the port Streamlit will run on
EXPOSE 10000

# Run the app using the startup script
CMD ["./start.sh"]
