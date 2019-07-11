FROM golang:1.11.9-alpine3.9
WORKDIR /go/src/app
RUN apk add --no-cache entr git gcc musl-dev
ENV GO111MODULE=on
ENV TERM=xterm-256color
EXPOSE 8080
COPY main.go go.mod data.csv ./
RUN go build main.go
CMD ["./main"]