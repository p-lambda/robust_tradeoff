FROM codalab/default-gpu
SHELL ["/bin/bash", "-c"]

LABEL maintainer="xie@cs.stanford.edu"

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y --no-install-recommends make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

RUN git clone git://github.com/yyuu/pyenv.git .pyenv

ENV HOME /
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

RUN pyenv install 3.7.0
RUN pyenv global 3.7.0
RUN pyenv rehash
RUN pip install --upgrade pip
ADD ./requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN apt-get install -y texlive-full
RUN apt-get install bc
