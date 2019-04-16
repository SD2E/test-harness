FROM gzynda/tacc-maverick-ml:latest

# we might need to remove all the LD_LIBRARY_PATH lines at some point
ENV LD_LIBRARY_PATH "/opt/conda/lib/:$LD_LIBRARY_PATH"
ENV LD_LIBRARY_PATH "/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH"
ENV LD_LIBRARY_PATH "/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH"
ENV HDF5_USE_FILE_LOCKING FALSE

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
RUN pip install gitpython
# we need a more modern version of pandas than is included in the base image
RUN pip uninstall --yes pandas
RUN pip install pandas==0.24.1

ADD scripts_for_automation/perovskite_test_harness.py /scripts_for_automation/
ADD scripts_for_automation/perovskite_model_run.py /scripts_for_automation/
ADD scripts_for_automation/perovskite_models_config.py /scripts_for_automation/
ADD harness/ /harness/
# Niall does not think it's a good idea to have the "ADD / /" line
# he says that line will probably "add your entire machine to the vm including os"
# But we can clearly see it's needed to import the version file in setup.py
ADD / /

ENV PYTHONPATH "${PYTHONPATH}:/scripts/"
ENV PYTHONPATH "${PYTHONPATH}:/test-harness/"
ENV PYTHONPATH "${PYTHONPATH}:/harness/"
ENV PYTHONPATH "${PYTHONPATH}:/"
