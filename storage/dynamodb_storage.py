import boto3
from boto3.dynamodb.conditions import Key
from typing import List, Dict
from .base import BaseStorage
import os
import hashlib
from decimal import Decimal
from datetime import datetime


class DynamoDBStorage(BaseStorage):
    def __init__(self, table_name: str = None):
        if table_name is None:
            table_name = os.getenv("DYNAMODB_TABLE", "maisa-YoloPredictions-Dev")

        # Use US East 2 (Ohio) region
        self.dynamodb = boto3.resource("dynamodb", region_name="us-east-2")
        self.table_name = table_name
        self.table = self.dynamodb.Table(table_name)

        # Initialize table if it doesn't exist
        self._ensure_table_exists()

    def _ensure_table_exists(self):
        """Create DynamoDB table if it doesn't exist"""
        try:
            # Check if table exists
            self.table.load()
            print(f"✅ DynamoDB table '{self.table_name}' already exists in us-east-2")
        except self.dynamodb.meta.client.exceptions.ResourceNotFoundException:
            print(f"📦 Creating DynamoDB table '{self.table_name}' in us-east-2...")
            self._create_table()

    def _create_table(self):
        """Create the DynamoDB table with proper schema"""
        try:
            table = self.dynamodb.create_table(
                TableName=self.table_name,
                KeySchema=[
                    {'AttributeName': 'PK', 'KeyType': 'HASH'},
                    {'AttributeName': 'SK', 'KeyType': 'RANGE'}
                ],
                AttributeDefinitions=[
                    {'AttributeName': 'PK', 'AttributeType': 'S'},
                    {'AttributeName': 'SK', 'AttributeType': 'S'},
                    {'AttributeName': 'label', 'AttributeType': 'S'},
                    {'AttributeName': 'score', 'AttributeType': 'N'}
                ],
                GlobalSecondaryIndexes=[
                    {
                        'IndexName': 'LabelIndex',
                        'KeySchema': [
                            {'AttributeName': 'label', 'KeyType': 'HASH'}
                        ],
                        'Projection': {'ProjectionType': 'ALL'},
                        'ProvisionedThroughput': {
                            'ReadCapacityUnits': 5,
                            'WriteCapacityUnits': 5
                        }
                    },
                    {
                        'IndexName': 'ScoreIndex',
                        'KeySchema': [
                            {'AttributeName': 'score', 'KeyType': 'HASH'}
                        ],
                        'Projection': {'ProjectionType': 'ALL'},
                        'ProvisionedThroughput': {
                            'ReadCapacityUnits': 5,
                            'WriteCapacityUnits': 5
                        }
                    }
                ],
                ProvisionedThroughput={
                    'ReadCapacityUnits': 5,
                    'WriteCapacityUnits': 5
                }
            )

            # Wait for table to be created
            table.wait_until_exists()
            self.table = table
            print(f"✅ DynamoDB table '{self.table_name}' created successfully in us-east-2")

        except Exception as e:
            print(f"❌ Failed to create DynamoDB table: {e}")
            raise

    def save_prediction(self, uid: str, original_image: str, predicted_image: str) -> None:
        """Save metadata for a prediction session"""
        timestamp = datetime.now().isoformat()

        item = {
            "PK": f"PRED#{uid}",
            "SK": "META",
            "uid": uid,
            "timestamp": timestamp,
            "original_image": original_image,
            "predicted_image": predicted_image
        }

        print(f"✅ Saving prediction metadata to {self.table_name}: {item}")
        try:
            self.table.put_item(Item=item)
        except Exception as e:
            print(f"❌ Failed to save prediction: {e}")
            raise

    def save_detection(self, prediction_uid: str, label: str, score: float, box: List[float]) -> None:
        """Save a single detected object"""
        detection_id = hashlib.md5(f"{label}-{score}-{box}".encode()).hexdigest()

        item = {
            "PK": f"PRED#{prediction_uid}",
            "SK": f"DETECT#{label}#{detection_id}",
            "prediction_uid": prediction_uid,
            "label": label,
            "score": Decimal(str(score)),
            "box": [Decimal(str(x)) for x in box]
        }

        print(f"✅ Saving detection to {self.table_name}: {item}")
        try:
            self.table.put_item(Item=item)
        except Exception as e:
            print(f"❌ Failed to save detection: {e}")
            raise

    def get_prediction(self, uid: str) -> Dict:
        """Retrieve full prediction session including metadata and all detections"""
        try:
            response = self.table.query(
                KeyConditionExpression=Key("PK").eq(f"PRED#{uid}")
            )

            items = response.get("Items", [])
            if not items:
                return None

            # Find metadata item
            meta = next((item for item in items if item["SK"] == "META"), None)
            if not meta:
                return None

            # Extract detection objects
            detections = []
            for item in items:
                if item["SK"].startswith("DETECT#"):
                    detections.append({
                        "id": item["SK"],  # Use SK as ID since we don't have auto-increment
                        "label": item["label"],
                        "score": float(item["score"]),  # Convert Decimal back to float
                        "box": [float(x) for x in item["box"]]  # Convert Decimal back to float
                    })

            return {
                "uid": uid,
                "timestamp": meta.get("timestamp"),
                "original_image": meta["original_image"],
                "predicted_image": meta["predicted_image"],
                "detection_objects": detections
            }

        except Exception as e:
            print(f"❌ Failed to get prediction: {e}")
            return None

    def get_predictions_by_label(self, label: str) -> List[Dict]:
        """Get all prediction sessions that include a detection with a specific label"""
        try:
            response = self.table.query(
                IndexName="LabelIndex",
                KeyConditionExpression=Key("label").eq(label)
            )

            items = response.get("Items", [])
            predictions = {}

            for item in items:
                if item.get("SK", "").startswith("DETECT#"):
                    pred_uid = item["PK"].split("#")[1]
                    if pred_uid not in predictions:
                        # Get metadata for this prediction to include timestamp
                        meta_response = self.table.get_item(
                            Key={"PK": f"PRED#{pred_uid}", "SK": "META"}
                        )
                        meta_item = meta_response.get("Item", {})

                        predictions[pred_uid] = {
                            "uid": pred_uid,
                            "timestamp": meta_item.get("timestamp")
                        }

            return list(predictions.values())

        except Exception as e:
            print(f"❌ Failed to get predictions by label: {e}")
            return []

    def get_predictions_by_score(self, min_score: float) -> List[Dict]:
        """Get all prediction sessions that include detections with score >= min_score"""
        try:
            # Note: This uses scan which is not efficient for large datasets
            # In production, consider using a different approach or GSI
            response = self.table.scan()
            items = response.get("Items", [])

            predictions = {}

            for item in items:
                if (item.get("SK", "").startswith("DETECT#") and
                        float(item.get("score", 0)) >= min_score):

                    pred_uid = item["PK"].split("#")[1]
                    if pred_uid not in predictions:
                        # Get metadata for this prediction to include timestamp
                        meta_response = self.table.get_item(
                            Key={"PK": f"PRED#{pred_uid}", "SK": "META"}
                        )
                        meta_item = meta_response.get("Item", {})

                        predictions[pred_uid] = {
                            "uid": pred_uid,
                            "timestamp": meta_item.get("timestamp")
                        }

            return list(predictions.values())

        except Exception as e:
            print(f"❌ Failed to get predictions by score: {e}")
            return []

    def get_prediction_image_path(self, uid: str) -> str:
        """Get the path to the predicted image file for a given prediction UID"""
        try:
            response = self.table.get_item(
                Key={"PK": f"PRED#{uid}", "SK": "META"}
            )

            item = response.get("Item")
            if not item:
                return None

            return item["predicted_image"]

        except Exception as e:
            print(f"❌ Failed to get prediction image path: {e}")
            return None