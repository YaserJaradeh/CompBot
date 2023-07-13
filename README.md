# Grizzly

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

A chatbot on top of the ORKG comparisons

## How to Run
You can run Grizzly in two ways. Docker and via Python.

### Docker
```bash
docker-compose up
```
This will build the image and run the container. The container will be accessible on port 4321 by default.

Check the [docker-compose.yml](docker-compose.yml) file for more details and customizations.

### Python
You need to create a virtual environment and install the requirements, then enable pre-commit scripts and run the app via uvicorn
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pre-commit install
uvicorn app.main:app --reload
```

## Roadmap
Some of the features that we are planning to add to Grizzly are:
- [ ] Add a web interface
- [ ] Include additional graph data in the comparison
- [ ] Add a CLI interface
- [ ] Support chart plotting
- [ ] Save feedbacks in the database
- [x] Support websockets for streaming the results

## License

This project is licensed under the terms of the [MIT](LICENSE) license.
