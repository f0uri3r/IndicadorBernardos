FROM python:3.9

RUN mkdir /srv/project/

COPY indicador_bernardos.py /srv/project/indicador_bernardos.py

COPY requirements.txt /srv/project/requirements.txt

COPY keys/ srv/project/keys/

WORKDIR /srv/project/

RUN mkdir output/

RUN pip3 install -r requirements.txt

EXPOSE 34617
