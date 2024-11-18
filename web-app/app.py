""" Flask server for web application - Project 4 """

import os
import logging
import requests
from flask import Flask, render_template, request, Blueprint
from dotenv import load_dotenv
from save_data import save_to_mongo
from get_statistics import get_statistics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


load_dotenv()
ml_base_url = os.getenv("ML_CLIENT_PORT")

def create_app():
    """Creates test app for pytest"""
    test_app = Flask(__name__)
    test_app.register_blueprint(main_bp)
    return test_app

# Define routes under the blueprint
@main_bp.route("/")
def main_page():
    """Render main page"""
    return render_template("main.html")


@main_bp.route("/statistics")
def statistics_page():
    """Render statistics page"""
    return render_template("statistics.html")


@main_bp.route("/classify", methods=["POST"])
def classify():
    """Call to ML API that classifies the user drawn number"""

    data = request.json
    response = None

    try:
        response = requests.post(ml_base_url + "/predict", json=data, timeout=30)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        logger.error("Request timed out")
        return {"error": "Request timed out"}, 500
    except requests.exceptions.RequestException as e:
        logger.error("An error occurred: %s", e)
        return {"error": f"An error occurred: {e}"}, 500

    if response.status_code == 200:
        return response.json()

    return {"error": "Classification failed"}, 500


@main_bp.route("/save-results", methods=["POST"])
def save_results():
    """Call to function that saves the result of classification"""

    data = request.json
    save_to_mongo(data)
    return "", 204


@main_bp.route("/get-stats", methods=["GET"])
def get_stats():
    """Call to function that retrieves app statistics"""

    return get_statistics()


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5001, debug=True)
