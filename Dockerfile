FROM gzynda/tacc-maverick-ml:latest

ENV LD_LIBRARY_PATH "/opt/conda/lib/:$LD_LIBRARY_PATH"
ENV LD_LIBRARY_PATH "/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH"
ENV LD_LIBRARY_PATH "/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH"
ENV LD_LIBRARY_PATH "/usr/local/cuda-8.0/targets/x86_64-linux/lib/stubs/:$LD_LIBRARY_PATH"

RUN apt-get update
RUN apt-get --assume-yes install libgl1
RUN apt-get --assume-yes install libx11-dev
RUN apt-get --assume-yes install xcb

RUN conda update scikit-learn

# todo: combine lines to make fewer layers
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
RUN pip install pyyaml
RUN pip install shap

ADD scripts_for_automation /scripts/
ADD scripts_for_automation /scripts/
ADD harness/ /harness/

ENV PYTHONPATH "${PYTHONPATH}:/scripts/"
ENV PYTHONPATH "${PYTHONPATH}:/test-harness/"
ENV PYTHONPATH "${PYTHONPATH}:/harness/"
ENV PYTHONPATH "${PYTHONPATH}:/"
