FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
# Add step to train the model
RUN python ./train_model.py  # This will train and save the model weights

CMD ["python", "./app.py"]
