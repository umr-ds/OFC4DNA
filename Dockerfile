FROM python:3.8-slim AS builder

MAINTAINER Peter Michael Schwarz "peter.schwarz@uni-marburg.de"

COPY . /optimize
WORKDIR /optimize
#RUN ls -la
#&& python setup.py install \
RUN apt-get update -y \
 && apt-get install --no-install-recommends -y apt-utils gcc build-essential ffmpeg google-perftools git \
 && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*
# RUN cd NOREC4DNA && git apply ../RulePatch.patch && git apply patch/silent_c_extension.patch && git apply patch/no_progressbar.patch && cd ..
#RUN pip3 install -r requirements.txt --no-cache-dir
##ENV VIRTUAL_ENV=/opt/venv
##RUN python3 -m venv $VIRTUAL_ENV
##ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN cd NOREC4DNA && pip3 install wheel && pip3 install -r requirements.txt --no-cache-dir && python3 setup.py install && cd ..
#RUN chmod +x setup.sh && .\setup.sh
RUN apt-get purge -y --auto-remove build-essential \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


# squash / reduce size
FROM scratch
COPY --from=builder / /
WORKDIR /optimize
##ENV VIRTUAL_ENV=/opt/venv
##ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENTRYPOINT ["python", "OptimizationSuite.py"]
