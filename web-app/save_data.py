""" Module for saving classification outputs and user inputs to mongodb """
import logging
from db_connect import connect_to_db
db = connect_to_db()
collection = db['num_classifications']
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_to_mongo(data):
    """Function to save data to mongo db"""
    try:
        document = {
            'intended_num': data['intendedNum'],
            'classified_num': data['classifiedNum'],
            'image_data': data['imageData'],
        }

        result = collection.insert_one(document)
        logger.info("Document saved to MongoDB with id: %s",
                    result.inserted_id)
        return True

    except KeyError as e:
        logger.error("Missing required field in data: %s", str(e))
        return False
    except TypeError as e:
        logger.error("Invalid data type: %s", str(e))
        return False
