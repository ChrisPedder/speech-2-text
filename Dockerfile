# Build image based on debian python 3.8 image
FROM python:3.8

COPY . .

RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential gcc \
                                        libsndfile1
RUN pip3 install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python3", "api_gateway.py"]
