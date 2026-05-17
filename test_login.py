import requests

try:
    resp = requests.post("https://quant-backend-w1lf.onrender.com/auth/login", json={
        "email": "kaushal@gmail.com",
        "password": "Kaushal@2004"
    }, timeout=10)
    print("kaushal@gmail.com login:", resp.status_code, resp.text[:100])
except Exception as e:
    print("Error:", e)

try:
    resp = requests.post("https://quant-backend-w1lf.onrender.com/auth/login", json={
        "email": "kaushalnandania086@gmail.com",
        "password": "Kaushal@2004"
    }, timeout=10)
    print("kaushalnandania086@gmail.com login:", resp.status_code, resp.text[:100])
except Exception as e:
    print("Error:", e)

