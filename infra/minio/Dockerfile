FROM golang:1.24-alpine AS build

WORKDIR /src
ADD --checksum=sha256:2f63e7c6867e13fe9dd3bd7280b2326bfd83232df60c40b250c4290c4e962a8a \
    https://github.com/minio/minio/archive/refs/tags/RELEASE.2025-07-23T15-54-02Z.tar.gz \
    /tmp/minio-source.tar.gz
RUN mkdir /src/minio && tar -xzf /tmp/minio-source.tar.gz \
    --strip-components=1 -C /src/minio
WORKDIR /src/minio
RUN mkdir -p /out && CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -tags kqueue -trimpath -o /out/minio .

FROM scratch
COPY --from=build /out/minio /minio
VOLUME ["/data"]
EXPOSE 9000 9001
ENTRYPOINT ["/minio"]
CMD ["server", "/data", "--console-address", ":9001"]
