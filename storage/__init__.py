import os

from .sqlite_storage import SQLiteStorage
from .dynamodb_storage import DynamoDBStorage
from .base import BaseStorage


def get_storage() -> BaseStorage:
    storage_type = os.getenv("STORAGE_TYPE", "sqlite").lower()

    if storage_type == "dynamodb":
        return DynamoDBStorage()
    elif storage_type == "sqlite":
        return SQLiteStorage()
    else:
        raise ValueError(f"Unsupported STORAGE_TYPE: {storage_type}")