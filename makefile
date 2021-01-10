
all:
	echo "No action."


init-venv:
	python3 -m venv env_honours
	source env_honours/bin/activate
	pip install ipykernel
	python -m ipykernel install --user --name=honours

jupyter:
	jupyter lab
