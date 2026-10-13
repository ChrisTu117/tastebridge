FROM python:3.12-slim
WORKDIR /app
COPY tastebridge ./tastebridge
ENV PORT=7860 QLOO_CACHE=/tmp/qloo-cache
EXPOSE 7860
USER 1000
CMD ["python", "-m", "tastebridge.server"]
