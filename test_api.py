import requests
import json

# Define the URL for your API endpoint
url = 'http://127.0.0.1:5000/predict'

# Sample data to send
data = [
    {"text": "The company's earnings exceeded expectations."},  # Positive
    {"text": "There was a loss reported last quarter."},      # Negative
    {"text": "The new product launch was okay."}              # Neutral
]


# Make the POST request to the API
response = requests.post(url, json=data)

# Print the status code and response
print("Status Code:", response.status_code)
print("Response JSON:", response.json())
