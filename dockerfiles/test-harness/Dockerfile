FROM gzynda/tacc-maverick-ml:latest

ENV LD_LIBRARY_PATH "/opt/conda/lib/:$LD_LIBRARY_PATH"
ENV LD_LIBRARY_PATH "/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH"
ENV LD_LIBRARY_PATH "/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH"

RUN apt-get update
RUN apt-get --assume-yes install libgl1
RUN apt-get --assume-yes install libx11-dev
RUN apt-get --assume-yes install xcb

RUN conda update scikit-learn

RUN pip install --upgrade pip
RUN pip install scikit-learn --upgrade
RUN pip install rfpimp
RUN pip install tabulate
RUN pip install lxml
RUN pip install bs4
RUN pip install hashids
RUN pip install BlackBoxAuditing
RUN pip install eli5
RUN pip install hyperas

#ADD protein-design /protein-design
#RUN cd /protein-design && python3 setup.py install
