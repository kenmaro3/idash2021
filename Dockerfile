FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive \
    SEAL_VERSION=3.6.5 \
    CMAKE_VERSION=3.19.5 \ 
    PYTHON_VERSION=3.6.9

RUN apt-get update && \
    apt-get install -y build-essential wget m4 file git vim libomp-dev libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev llvm libncurses5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
RUN        apt-get clean && rm -rf /var/lib/apt-get/lists/*

# CMake Installation 
WORKDIR /include
RUN wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}.tar.gz && tar -zxvf ./cmake-${CMAKE_VERSION}.tar.gz && \
    cd cmake-${CMAKE_VERSION} && \
    ./bootstrap --prefix=/opt/cmake -- -DCMAKE_BUILD_TYPE:STRING=Release -DCMAKE_USE_OPENSSL=OFF && \
    make -j4 && make install && make clean \
    && rm  /include/cmake-${CMAKE_VERSION}.tar.gz


# Install Python by pyenv and pyenv-virtualenv 
SHELL ["/bin/bash", "-c"]


# SEAL installation
RUN git clone https://github.com/kenmaro3/SEAL.git && \
    cd SEAL && \
    git checkout -b ckks_coeff_365 origin/ckks_coeff_365 &&\
    /opt/cmake/bin/cmake . -DCMAKE_BUILD_TYPE=Release -DSEAL_BUILD_EXAMPLES=OFF -DSEAL_BUILD_TESTS=OFF -DSEAL_BUILD_BENCH=OFF -DSEAL_BUILD_DEPS=ON -DSEAL_USE_INTEL_HEXL=ON -DSEAL_USE_MSGSL=ON -DSEAL_USE_ZLIB=ON -DSEAL_USE_ZSTD=ON -DBUILD_SHARED_LIBS=OFF -DSEAL_BUILD_SEAL_C=ON -DSEAL_USE_CXX17=ON -DSEAL_USE_INTRIN=ON &&\
    make -j4 && make install &&\
    echo export PATH="$PATH:/opt/cmake/bin" > ~/.bashrc 

# python installation
RUN git clone https://github.com/pyenv/pyenv.git ~/.pyenv &&\
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc &&\
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc &&\
    echo "export PYENV_VIRTUALENV_DISABLE_PROMPT=1" >> ~/.bashrc &&\
    echo "export SEAL_VERSION="${SEAL_VERSION} >> ~/.bashrc &&\
    echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n eval "$(pyenv init -)"\nfi' >> ~/.bashrc &&\
    source ~/.bashrc &&\
    pyenv install ${PYTHON_VERSION} &&\
    git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv &&\
    echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc && source ~/.bashrc

COPY requirements.txt /from_local/

WORKDIR /from_local

RUN  source ~/.bashrc &&\
    pyenv virtualenv ${PYTHON_VERSION} myenv && pyenv global ${PYTHON_VERSION} && source activate myenv

RUN ~/.pyenv/versions/myenv/bin/pip install --upgrade pip setuptools && ~/.pyenv/versions/myenv/bin/pip install -r requirements.txt
