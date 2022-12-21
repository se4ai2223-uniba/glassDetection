# ---- Base python ----
FROM python:3.9-slim

# Install dependencies
COPY ./app_frontend//requirements_frontend.txt /app_frontend/requirements_frontend.txt
RUN pip install -r /app_frontend/requirements_frontend.txt


RUN pip install -U pip && pip install --no-cache-dir --upgrade -r /app_frontend/requirements_frontend.txt

# Copy source code
COPY . /app_frontend

# Expose port
EXPOSE 8001

WORKDIR /app_frontend

CMD ["uvicorn", "app_frontend.interface:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]