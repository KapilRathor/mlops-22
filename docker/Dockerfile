FROM ubuntu:latest
FROM python:3.8.1
COPY ./*.py /exam/
COPY ./requirements.txt /exam/requirements.txt
RUN pip3 install --no-cache-dir -r /exam/requirements.txt
WORKDIR /exam
CMD ["python3", "./plot_graphs.py", ]