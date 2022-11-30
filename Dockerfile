FROM python:3.9

# Install dependencies
COPY ./requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt


RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install dlib==19.22
# Copy source code
COPY . /app

# Expose port
EXPOSE 8000

WORKDIR /app

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]