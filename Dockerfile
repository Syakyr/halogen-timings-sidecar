# Drop-in Halogen image with a timings sidecar in front of /v1.
# Same devices, same env, same published port 8731.
ARG HALOGEN_IMAGE=ghcr.io/peonist-ai/halogen-flash-server:0.11.9
FROM ${HALOGEN_IMAGE}

COPY proxy.py wrap.sh /opt/halogen-sidecar/
RUN chmod +x /opt/halogen-sidecar/wrap.sh /opt/halogen-sidecar/proxy.py

# Original CMD (`all` / `engine` / `api` / `bench` / `sweep`) is preserved
# as arguments to the new entrypoint.
ENTRYPOINT ["/opt/halogen-sidecar/wrap.sh"]
