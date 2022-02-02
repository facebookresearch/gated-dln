import sys

from xplogger.experiment_manager.store.mongo import MongoStore

sys.path.append("/private/home/sodhani/projects/abstraction_by_gating/")

from omegaconf import OmegaConf
from xplogger.logger.mongo import Logger as MongoLogger


def get_mongo_store():

    collection_name = "abstraction_by_gating"
    mongo_store = MongoStore(
        config={
            "host": "localhost",
            "port": 27017,
            "db": "project",
            "collection_name": collection_name,
        }
    )
    return mongo_store


def get_mongo_logger():
    mongo_config = OmegaConf.load(
        "/private/home/sodhani/projects/abstraction_by_gating/config/logbook/xplogger.yaml"
    ).mongo_config

    mongo_logger = MongoLogger(mongo_config)
    return mongo_logger
