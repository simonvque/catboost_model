import requests

url = "http://127.0.0.1:5000/predict"
payload = {
    "Temperature": 30,
    "Humidity": 85,
    "Precipitation": 99,
    "Signal Strength": -89,
    "Packet Loss": 9
}
response = requests.post(url, json=payload)
print(response.json())
