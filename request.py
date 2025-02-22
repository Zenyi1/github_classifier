import requests

response = requests.post(
    "https://github-classifier.onrender.com/predict",  # Add the /predict endpoint
    json={"image_url": "https://images.pexels.com/photos/6477261/pexels-photo-6477261.jpeg"}
)

# Add error handling to see what's going wrong
print(f"Status Code: {response.status_code}")
print(f"Response Text: {response.text}")
try:
    print(f"JSON Response: {response.json()}")
except requests.exceptions.JSONDecodeError as e:
    print(f"Failed to decode JSON: {e}")