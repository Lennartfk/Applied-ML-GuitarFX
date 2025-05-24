import requests

url = "http://localhost:8000/models/svm"

response = requests.post(url)

print(response.status_code)
print(response.json())
