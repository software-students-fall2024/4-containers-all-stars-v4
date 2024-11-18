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

- Using Docker

1. If you haven't already, install the `Docker Desktop`.
2. Make sure Docker Desktop is open.
3. Go to the main directory of the project and run `docker-compose up --build` in the terminal.

- Using Virtual Environments

1. If you haven't already, install `pipenv` using the command: `pip install pipenv`.
2. Create two terminals, one that's in the web-app subdirectory and another that's in the machine-learning-client subdirectory. Do the next insturctions for both terminals.
3. Create virtual environments by running `pipenv install`
4. Activate the virtual environment: `pipenv shell`.
5. Install additional dependencies: `pip install -r requirements.txt`
6. Run the app: `python app.py`

## .Env files instructions

1. Go to the main directory of the project and create a `.env` file: macOs/Linux - `touch .env`, Windows - `New-Item -Path . -Name ".env" -ItemType "File"`
2. Copy and paste the contents below:

MONGO_CXN_STRING=mongodb+srv://inp9194:3lVvTkvkz3lDh1z1@allstars.ea528.mongodb.net/?retryWrites=true&w=majority&appName=allstars
ML_CLIENT_PORT=http://localhost:5002
MONGO_USERNAME=inp9194
MONGO_PASSWORD=3lVvTkvkz3lDh1z1

