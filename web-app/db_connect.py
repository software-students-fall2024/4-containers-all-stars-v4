""" Module for connecting to mongodb """
# import certifi
import os
import logging
import pymongo
from dotenv import load_dotenv


def connect_to_db():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    load_dotenv()
    mongo_cxn = os.getenv('MONGO_CXN_STRING')
    # client = pymongo.MongoClient(mongo_cxn, tlsCAFile=certifi.where())
    client = pymongo.MongoClient(mongo_cxn)

    db = client['project4']
    return db
