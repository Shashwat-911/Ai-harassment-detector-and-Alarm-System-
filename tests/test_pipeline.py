import base64
import json
import logging
import cv2
import numpy as np
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def test_analyze_endpoint():
    # 1. Create a dummy 640x480 black image
    logger.info("Generating 640x480 dummy black frame...")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 2. Encode this frame to base64 jpeg format
    success, buffer = cv2.imencode('.jpg', frame)
    if not success:
        logger.error("Failed to encode frame to JPEG.")
        return

    encoded_frame = base64.b64encode(buffer).decode('utf-8')

    # 3. Define payload
    url = "http://localhost:8000/api/v1/analyze"
    payload = {
        "frame_id": "test_001",
        "base64_image": encoded_frame,
        "camera_id": "lab_cam_01"
    }

    # 4. Send POST request
    logger.info(f"Sending POST request to {url} ...")
    try:
        response = requests.post(url, json=payload)
        
        # 5. Print status code and full response JSON
        logger.info(f"Status Code: {response.status_code}")
        try:
            logger.info(f"Response JSON: {json.dumps(response.json(), indent=4)}")
        except ValueError:
            logger.info(f"Response Text: {response.text}")
            
    except requests.exceptions.ConnectionError:
        logger.error(
            "ConnectionError: The server is not running or unreachable. "
            "Please ensure that the FastAPI server is started on localhost:8000."
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    test_analyze_endpoint()
