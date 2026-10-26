FROM python:3.12.13-slim
WORKDIR /app
COPY pyproject.toml ./
COPY NOTICE.md ./
COPY licenses ./licenses
COPY src ./src
RUN pip install --no-cache-dir .
USER 65534:65534
ENTRYPOINT ["python", "-m", "opencv_deploy_proof.cli"]
