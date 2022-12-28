# ---- Base python ----
FROM python:3.9-slim

# Install dependencies
COPY ./app_frontend/requirements_frontend.txt /app/requirements_frontend.txt
RUN pip install -r /app/requirements_frontend.txt


RUN pip install -U pip && pip install --no-cache-dir --upgrade -r /app/requirements_frontend.txt

# Copy source code
COPY . /app

# Expose port
EXPOSE 8001

WORKDIR /app

CMD ["uvicorn", "app_frontend.interface:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]