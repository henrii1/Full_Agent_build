install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	python -m pytest -vv --cov=webapp tests/test_*.py

format:	
	black webapp/*.py tests/*.py

lint:
	pylint --disable=R,C --extension-pkg-whitelist='pydantic' webapp/main.py --ignore-patterns=tests/test_.*?py  webapp/*.py


refactor: format lint

deploy:  #deploy step to be changed after creating the AWS container repo
	

all: install lint test format