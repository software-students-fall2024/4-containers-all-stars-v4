""" Flask server for web application - Project 4 """

import os
import logging
import requests
from flask import Flask, render_template, request
from dotenv import load_dotenv
from save_data import save_to_mongo
from get_statistics import get_statistics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

load_dotenv()
ml_base_url = os.getenv('ML_CLIENT_PORT')


@app.route('/')
def main_page():
    """ render main page """
    return render_template('main.html')


@app.route('/statistics')
def statistics_page():
    """ render statistics page """
    return render_template('statistics.html')


@app.route('/classify', methods=['POST'])
def classify():
    """ call to ml api that classifies user drawn num """

    data = request.json

    try:
        response = requests.post(
            ml_base_url + '/predict', json=data, timeout=30)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        print("Request timed out")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

    if response.status_code == 200:
        return response.json()

    return {"error": "Classification failed"}, 500


@app.route('/save-results', methods=['POST'])
def save_results():
    """ call to function that saves result of classification """

    data = request.json
    save_to_mongo(data)
    return '', 204


@app.route('/get-stats', methods=['GET'])
def get_stats():
    """ call to function that retrieves app statistics """

    return get_statistics()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
