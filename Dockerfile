FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install matplotlib numpy pandas scikit-learn seaborn streamlit

EXPOSE 8000

CMD ["streamlit", "run", "heart.py"]