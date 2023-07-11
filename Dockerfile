FROM python:3.10
LABEL maintainer="Yaser Jaradeh <Yaser.Jaradeh@tib.eu>"

WORKDIR /grizzly

RUN pip install poetry

COPY ./pyproject.toml ./poetry.lock* /grizzly/

RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /grizzly/requirements.txt

COPY . /grizzly

CMD ["gunicorn", "app.main:app", "--workers", "1",  "--timeout", "0", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:4321", "--access-logfile=-", "--error-logfile=-"]
