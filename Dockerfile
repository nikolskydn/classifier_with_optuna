FROM python:3.10
WORKDIR /app
RUN apt-get update -y
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
ENTRYPOINT ["./entrypoint.sh"]
