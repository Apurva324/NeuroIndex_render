FROM python:3.11-slim

RUN useradd -m -u 1000 user

WORKDIR /home/user/app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user . .

USER user

RUN chmod +x start.sh

ENV VECTORDB_HOST=0.0.0.0

EXPOSE 10000

CMD ["./start.sh"]
