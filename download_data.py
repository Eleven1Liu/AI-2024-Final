import os
import requests
import time

from datetime import datetime
import pandas as pd


URL = "https://tcgbusfs.blob.core.windows.net/dotapp/youbike/v2/youbike_immediate.json"
INTERVAL = 300
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


while True:
    r = requests.get(URL)

    # convert time
    gmt_time = datetime.strptime(r.headers['Date'], "%a, %d %b %Y %H:%M:%S %Z")
    formatted_time = gmt_time.strftime("%Y%m%d%H%M%S")
    
    if r.status_code == 200:
        df = pd.DataFrame(r.json())
        df.to_csv(f"{DATA_DIR}/{formatted_time}.csv", index=False)
    else:
        with open("error.log", "a") as f:
            f.write("[{formatted_time}] Failed to retrieve data from the url.")
    
    time.sleep(INTERVAL)
