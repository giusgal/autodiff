FROM alpine:edge

RUN apk add --no-cache \
    git \
    cmake \
    make \
    g++ \
    build-base \
    eigen-dev

WORKDIR /app

RUN git clone https://github.com/giusgal/autodiff.git

WORKDIR /app/autodiff
RUN mkdir build

WORKDIR /app/autodiff/build
RUN cmake .. \
 && make

CMD ["/bin/sh"]
