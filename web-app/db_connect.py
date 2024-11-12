""" Module for connecting to mongodb """
# import certifi
import os
import pymongo
from dotenv import load_dotenv


def connect_to_db():
    """ Configures logger and connects to db """
    load_dotenv()
    mongo_cxn = os.getenv('MONGO_CXN_STRING')
    # client = pymongo.MongoClient(mongo_cxn, tlsCAFile=certifi.where())
    client = pymongo.MongoClient(mongo_cxn)

    db = client['project4']
    return db