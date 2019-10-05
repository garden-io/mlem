FROM golang:1.11.9-alpine3.9 AS builder
WORKDIR /go/src/app
RUN apk add --no-cache git gcc musl-dev
ENV GO111MODULE=on
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -mod vendor -o main -ldflags '-w'

FROM alpine
# GLIBC
ENV \
    ALPINE_GLIBC_URL="https://github.com/sgerrand/alpine-pkg-glibc/releases/download/2.21-r2/" \
    GLIBC_PKG="glibc-2.21-r2.apk" \
    GLIBC_BIN_PKG="glibc-bin-2.21-r2.apk"
RUN \
    apk add --update -t deps wget ca-certificates openssl \
    && apk add --update -t openssl \ 
    && cd /tmp \
    && wget ${ALPINE_GLIBC_URL}${GLIBC_PKG} ${ALPINE_GLIBC_URL}${GLIBC_BIN_PKG} \
    && apk add --allow-untrusted ${GLIBC_PKG} ${GLIBC_BIN_PKG} \
    && /usr/glibc/usr/bin/ldconfig /lib /usr/glibc/usr/lib \
    && echo 'hosts: files mdns4_minimal [NOTFOUND=return] dns mdns4' >> /etc/nsswitch.conf \
    && apk del --purge deps \
    && rm /tmp/* /var/cache/apk/*
ENV TERM=xterm-256color
EXPOSE 8080
COPY --from=builder /go/src/app/gotty /usr/bin/
COPY --from=builder /go/src/app/main ./
CMD gotty ./main