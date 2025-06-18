from abc import ABC, abstractmethod
from typing import List, Dict


class BaseStorage(ABC):
    @abstractmethod
    def save_prediction(self, uid: str, original_image: str, predicted_image: str) -> None:
        """
        Save metadata for a prediction session.
        """
        pass

    @abstractmethod
    def save_detection(self, prediction_uid: str, label: str, score: float, box: List[float]) -> None:
        """
        Save a single detected object (label, score, bounding box) for a given prediction session.
        """
        pass

    @abstractmethod
    def get_prediction(self, uid: str) -> Dict:
        """
        Retrieve full prediction session including metadata and all detections.
        """
        pass

    @abstractmethod
    def get_predictions_by_label(self, label: str) -> List[Dict]:
        """
        Get all prediction sessions that include a detection with a specific label.
        """
        pass

    @abstractmethod
    def get_predictions_by_score(self, min_score: float) -> List[Dict]:
        """
        Get all prediction sessions that include detections with score >= min_score.
        """
        pass

    @abstractmethod
    def get_prediction_image_path(self, uid: str) -> str:
        """
        Get the path to the predicted image file for a given prediction UID.
        """
        pass