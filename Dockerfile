FROM python:3.9

WORKDIR /HW1_project

COPY . /HW1_project

RUN pip install numpy pandas matplotlib scikit-learn tensorflow keras einops dataclasses

CMD [ "bash", "start.sh" ]
