FROM python:3.11.15-slim-trixie
WORKDIR /app
RUN pip install uv
COPY pyproject.toml uv.lock README.md ./
COPY ./src ./src
RUN uv sync --no-dev --frozen
EXPOSE 8000
CMD ["uv","run","uvicorn","rag_porfolio.app:app","--host","0.0.0.0","--port","8000"]