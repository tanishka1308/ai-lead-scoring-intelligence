# Use a stable Python 3.11 runtime
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable live log streaming
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install build essentials for C extensions if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifest and install packages
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy all application code into container
COPY . /app

# Ensure directories for artifacts and app database exist
RUN mkdir -p /app/artifacts /app/app

# Render passes the port via the $PORT environment variable
ENV PORT=8501
EXPOSE 8501

# Run the Streamlit dashboard bound to 0.0.0.0 and the dynamically assigned port
CMD streamlit run dashboard/app.py --server.port=${PORT} --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false