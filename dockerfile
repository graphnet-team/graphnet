# Reference image (large), for missing components.
FROM icecube/icetray:combo-stable as combo

# Base image.
FROM icecube/icetray:combo-stable-slim as main

# Argument(s).
ARG HARDWARE=cpu

# Copy over missing libraries in slim.
COPY --from=combo /usr/lib/x86_64-linux-gnu/libboost_chrono.so.1.71.0 /usr/lib/x86_64-linux-gnu/
COPY --from=combo /usr/lib/x86_64-linux-gnu/libboost_atomic.so.1.71.0 /usr/lib/x86_64-linux-gnu/

# Install pip.
RUN wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py && rm get-pip.py

# Updating python packages. Specific to `icecube/icetray:combo*`.
RUN pip install --upgrade pip && \
    pip install wheel setuptools==59.5.0 && \
    pip install --upgrade astropy && \
    pip install --ignore-installed PyYAML

# Install GraphNeT and required dependencies.
# -- Via pip
#RUN pip install -r https://raw.githubusercontent.com/com/graphnet-team/graphnet/main/requirements/torch_${HARDWARE}.txt
#RUN pip install -e git+https://github.com/icecube/graphnet.git#egg=graphnet[develop,torch]

# -- Using current source
WORKDIR /root/graphnet/
ADD . /root/graphnet/

RUN pip install -r requirements/torch_${HARDWARE}.txt
RUN pip install -e .[develop,torch]

# Create missing alias(es) in slim.
RUN echo 'alias python="python3"' >> ~/.bashrc

# Stylise command line prompt
RUN echo 'PS1="🐳 \[\033[38;2;86;138;242m\]graphnet@\h \[\033[0m\]❯ \[\033[0;34m\]\w \[\033[0m\]\$ "' >> ~/.bashrc
RUN echo 'PS2="\[\033[38;5;236m\]❯\[\033[38;5;239m\]❯\[\033[0m\]❯ "' >> ~/.bashrc

# Default command for executing container.
CMD [ "/bin/bash" ]