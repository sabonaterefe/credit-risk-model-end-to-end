FROM python:3.10-slim

# Set environment variable to suppress pip root warning
ENV PIP_ROOT_USER_ACTION=ignore

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Fix line endings and make start.sh executable
RUN sed -i 's/\r$//' start.sh && chmod +x start.sh

# Optional: create a non-root user (best practice)
RUN adduser --disabled-password --gecos '' appuser
USER appuser

# Expose the port used by Streamlit
EXPOSE 10000

# Run the app
CMD ["./start.sh"]
