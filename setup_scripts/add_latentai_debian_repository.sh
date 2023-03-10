#!/bin/bash
if ! command -v curl &> /dev/null; then
    echo "Curl not found, installing ..." && \
        apt update && \
        apt install -y --no-install-recommends curl
fi

apt list ca-certificates -a | grep installed &> /dev/null

[ "$?"!="0" ] && apt update && apt install -y --no-install-recommends ca-certificates

[ ! -f /etc/apt/trusted.gpg.d/latentai.gpg ] && curl --output /etc/apt/trusted.gpg.d/latentai.gpg https://apt-release.latentai.io/latentai.gpg

[ ! -f /etc/apt/sources.list.d/latentai-dev-stable.list ] && echo "deb https://apt-release.latentai.io/stable stable main" > /etc/apt/sources.list.d/latentai-dev-stable.list

[ "$?"="0" ] && apt update && \
    echo "Run apt install latentai-<package-name>"
