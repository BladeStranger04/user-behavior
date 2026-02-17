install:
	pip install -r requirements.txt

data:
	python generate_data.py

train:
	python run.py

ui:
	mlflow ui