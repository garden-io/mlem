FROM golang:1.11.9-alpine3.9 AS builder
WORKDIR /go/src/app
RUN apk add --no-cache git gcc musl-dev
ENV GO111MODULE=on
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -mod vendor -o main -ldflags '-w' -gcflags "all=-N -l"

FROM scratch
EXPOSE 8080
COPY --from=builder /go/src/app/main ./
# CMD /main
ENTRYPOINT ["/main"]