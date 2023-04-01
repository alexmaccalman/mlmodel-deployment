# docker build -t myapi .
# docker run -p 8080:8080 myapi
FROM python:3.8-slim

WORKDIR /app

COPY ./app /app

RUN pip install -r requirements.txt

EXPOSE 8080

CMD ["python", "app.py"]
