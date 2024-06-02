import argparse
import requests
import base64

# Create the parser
parser = argparse.ArgumentParser(description="Send a .obj file to an endpoint.")

# Add the arguments
parser.add_argument("path", metavar="path", type=str, help="the path to the .obj file")
parser.add_argument("prompt", metavar="prompt", type=str, help="the prompt to send")

# Parse the arguments
args = parser.parse_args()

# Define the URL of the endpoint
url = "http://localhost:8094/validate/"

# Open the .obj file in binary mode
with open(args.path, "rb") as f:
    # Read the file data
    file_data = f.read()

# Encode the binary data as a base64 string
file_data_base64 = base64.b64encode(file_data).decode('utf-8')

# Define the data to send to the endpoint
data = {
    "prompt": args.prompt,
    "data": file_data_base64,
}

# Send a POST request to the endpoint
response = requests.post(url, json=data)

# Print the response
print(response.json())