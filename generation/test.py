import requests

# Define the URL of the endpoint
url = "http://localhost:8094/validate/"

# Open the .obj file in binary mode
with open("test.obj", "rb") as f:
    # Read the file data
    file_data = f.read()

# Define the data to send to the endpoint
data = {
    "prompt": "pink bicycle",
    "data": file_data,
}

# Send a POST request to the endpoint
response = requests.post(url, json=data)

# Print the response
print(response.json())