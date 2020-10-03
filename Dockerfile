FROM python

WORKDIR /project
COPY . /project
RUN pwd
EXPOSE 80

RUN pip install -r requirements.txt
CMD python app.py