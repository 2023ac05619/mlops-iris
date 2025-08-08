FROM python:3.10-slim

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code and the pre-trained models
COPY ./main.py /app/main.py
COPY ./api /app/api
COPY ./db /app/db
COPY ./models /app/models
COPY ./src /app/src
# COPY ./train /app/train
# COPY ./config /app/config

EXPOSE 5001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5001"]
