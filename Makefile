install:
	sudo apt update
	sudo apt install pipx
	pipx ensurepath
	pipx install poetry
	poetry install --no-root

env:
	poetry shell

test:
	python -m pytest -vv tests/test_*.py

format:	
	black crewai/*.py tests/*.py langchain/*.py llamaindex/*.py app.py

lint:
	pylint --disable=R,C --extension-pkg-whitelist='pydantic' app.py --ignore-patterns=tests/test_.*?py  langchain/*.py \
		llamaindex/*.py crewai/*.py


refactor: format lint

deploy:  #deploy step to be changed after creating the AWS container repo
	

all: install lint test format