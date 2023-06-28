FROM python:3.10

RUN apt-get update && \
    apt-get install -y git

RUN pip install numpy pandas matplotlib streamlit plotly
RUN pip install sktime[all_extras]
