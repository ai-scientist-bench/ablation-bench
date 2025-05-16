FROM python:3.11

ENV TZ=Etc/UTC
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /

# Install swe-rex for faster startup
RUN pip install pipx
RUN pipx install swe-rex
RUN pipx ensurepath
ENV PATH="$PATH:/root/.local/bin/"

RUN git config --global user.name "Your Name"
RUN git config --global user.email "you@example.com"
RUN mkdir -p repo && cd /repo && git init && touch empty && git add empty && git commit -m "init"

SHELL ["/bin/bash", "-c"]
