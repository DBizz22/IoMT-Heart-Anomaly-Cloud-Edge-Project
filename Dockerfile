FROM rayproject/ray:2.46.0-py311

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

#TODO: modify copy to only essential files
COPY . /app

ENV PYTHONUNBUFFERED=1
ENV SERVE_HOST=0.0.0.0
ENV SERVE_PORT=8000

# Default command is inference. RayJob overrides command for training.
CMD ["python", "prediction_API.py"]
