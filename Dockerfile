FROM python:3.10.6-buster

RUN apt update
RUN apt install -y cmake libgl1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

#allows running on local and GCP
CMD streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT
