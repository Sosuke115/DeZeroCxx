FROM centos:centos7

RUN yum update -y 

RUN yum install -y \
    gperf \
    golang \
    ruby \
    libuuid-devel \
    libxml2-devel \
    wget \
    which \
    clang \
    make \
    valgrind \
    git 

# バージョン指定してCMakeをインストール
RUN mkdir -p /tmp/cmake && \
    pushd /tmp/cmake && \
    wget 'https://cmake.org/files/v3.25/cmake-3.25.0-linux-aarch64.sh' && \
    bash cmake-3.25.0-linux-aarch64.sh --prefix=/usr/local --exclude-subdir && \
    popd && \
    rm -rf /tmp/cmake

# GCC
RUN yum install -y \
    centos-release-scl
RUN yum install -y \
    devtoolset-10-gcc*
ENV PATH="/opt/rh/devtoolset-10/root/usr/bin:$PATH"
RUN source scl_source enable devtoolset-10

# boost
RUN cd /home && wget http://downloads.sourceforge.net/project/boost/boost/1.81.0/boost_1_81_0.tar.gz \
    && tar xfz boost_1_81_0.tar.gz \
    && rm boost_1_81_0.tar.gz \
    && cd boost_1_81_0 \
    && ./bootstrap.sh --prefix=/usr/local --with-libraries=program_options \
    && ./b2 install \
    && cd /home \
    && rm -rf boost_1_81_0

# NumCpp
RUN cd /home && git clone https://github.com/dpilger26/NumCpp.git && \
    cd NumCpp && mkdir build && cd build && \
    cmake -DNUMCPP_NO_USE_BOOST=ON .. && \
    cmake --build . --target install

RUN yum clean all

# Build directory
RUN mkdir -p /src
WORKDIR /src