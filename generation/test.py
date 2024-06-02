# import requests
# import base64

# # Define the URL of the endpoint
# url = "http://localhost:8094/validate/"

# # Open the .obj file in binary mode
# with open("test.obj", "rb") as f:
#     # Read the file data
#     file_data = f.read()

# # Encode the binary data as a base64 string
# file_data_base64 = base64.b64encode(file_data).decode('utf-8')

# # Define the data to send to the endpoint
# data = {
#     "prompt": "pink bicycle",
#     "data": file_data_base64,
# }

# # Send a POST request to the endpoint
# response = requests.post(url, json=data, timeout=300)

# # Print the response
# print(response.json())

import argparse
import requests
from time import time

# Create the parser
parser = argparse.ArgumentParser(description="Send a prompt to an endpoint.")

# Add the arguments
parser.add_argument("mode", metavar="mode", type=int, help="the mode to send")

# Parse the arguments
args = parser.parse_args()

# Define the URL of the endpoint
url = "http://localhost:8093/generate/"

# Define the data to send to the endpoint
data = {
    "prompt": "pink bicycle",
    "mode": args.mode or 1
}

# Send a POST request to the endpoint
start_time = time()
response = requests.post(url, data=data, timeout=600)

# Print the response
print(f"[INFO] It took: {(time() - start_time) / 60.0} min")