import requests

url = "http://localhost:8000/predict"

response = requests.post(url)

print(response.status_code)
print(response.json())
