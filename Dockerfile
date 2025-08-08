FROM python:3.10-slim

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code and the pre-trained models
COPY ./main.py /app/main.py
COPY ./app /app/app
COPY ./src /app/src
COPY ./config /app/config
COPY ./models /app/models

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
