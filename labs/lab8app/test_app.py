# lab8_flask/test_app.py

import requests

# 1) Update this URL if your server is bound elsewhere
URL = "http://127.0.0.1:8000/predict"

# 2) Payload must include all 13 features, 
#    using the underscore‐safe name for the slash column
payload = {
    "alcohol": 13.20,
    "malic_acid": 1.78,
    "ash": 2.14,
    "alcalinity_of_ash": 11.2,
    "magnesium": 100.0,
    "total_phenols": 2.65,
    "flavanoids": 2.76,
    "nonflavanoid_phenols": 0.26,
    "proanthocyanins": 1.28,
    "color_intensity": 4.38,
    "hue": 1.05,
    "od280_od315_of_diluted_wines": 3.40,
    "proline": 1050.0
}

def main():
    resp = requests.post(URL, json=payload)
    print("Status:", resp.status_code)
    try:
        data = resp.json()
        print("Body:", data)
    except ValueError:
        print("Non-JSON response:", resp.text)

if __name__ == "__main__":
    main()
