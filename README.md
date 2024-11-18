![Lint-free](https://github.com/software-students-fall2024/4-containers-all-stars-v4/actions/workflows/lint.yml/badge.svg)
![ML-Client CI Workflow](https://github.com/software-students-fall2024/4-containers-all-stars-v4/actions/workflows/ml-client.yml/badge.svg)
![Web-App CI Workflow](https://github.com/software-students-fall2024/4-containers-all-stars-v4/actions/workflows/web-app.yml/badge.svg)

# Containerized Digit Classification App

## Description

This project is an interactive digit recognition application. Users can draw numbers on the UI, which are instantly classified by a Convolutional Neural Network (CNN) trained on the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset. The application also has a statistics dashboard where users can view the model's performance. The system also stores all user inputs in MNIST-compatible format, creating a valuable dataset for future potential model training.

## Team Members

- [Obinna Nwaokoro](https://www.github.com/ocnwaokoro)
- [Marko Todorovic](https://github.com/mtodorovic27)
- [Sanskriti Gupta](https://github.com/sanskritig08)
- [Ian Pompliano](https://www.github.com/ianpompliano)

## Configuration Instructions

# There are two ways to configure and run all parts of the project

- Using Docker (easiest)

    1. If you haven't already, install the `Docker Desktop`.
    2. Make sure Docker daemon is running.
    3. Go to the main directory of the project and run `docker compose up --build` in the terminal.
    4. Enter the local address where the `web-app` container is running. This should be configured to port 5001.

- Using Virtual Environments

    1. If you haven't already, install `pipenv` using the command: `pip install pipenv`.
    2. Create two terminals. Navigate to `web-app` in one and `machine-learning-client` in the other.
    3. Create virtual environments by running `pipenv install` in both subdirectories.
    4. Activate the virtual environments in both terminals using `pipenv shell`.
    5. Run the flask apps for both `web-app` and `machine-learning-client`. The command is the same in both subdirectories.: `python app.py`
    6. If needed, install additional dependencies: `pip install -r requirements.txt`.

## .Env files instructions

Create a `.env` file in the main directory (same level as this README file)

