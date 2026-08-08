import requests
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("TWELVE_DATA_API_KEY")

url = "https://api.twelvedata.com/symbol_search"

params = {
    "symbol": "TCS",
    "apikey": api_key
}

response = requests.get(url, params=params)

print(response.json())