import argparse
import os
import requests
import time

from datetime import datetime
from socket import error as SocketError
import pandas as pd


URL = "https://tcgbusfs.blob.core.windows.net/dotapp/youbike/v2/youbike_immediate.json"


def download_data(data_dir, interval):
    try:
        r = requests.get(URL)
        gmt_time = datetime.strptime(r.headers['Date'], "%a, %d %b %Y %H:%M:%S %Z")
        formatted_time = gmt_time.strftime("%Y%m%d%H%M%S")
        
        if r.status_code == 200:
            df = pd.DataFrame(r.json())
            df.to_csv(f"{data_dir}/{formatted_time}.csv", index=False)
        else:
            # log connection error (404, 500)
            with open("error.log", "a") as f:
                f.write("[{formatted_time}] Failed to retrieve data from the url.")
        
    except SocketError as e:
        # socket error with out status code, pass and sleep for a while
        print("Socket error! Sleep for 5 minutes.")
        time.sleep(300)
        pass
    time.sleep(interval)
    

def main():
    
    parser = argparse.ArgumentParser(add_help=False)

    # load params from config file
    parser.add_argument("-d", "--data_dir", type=str, default="data", help="Path to the data directory.")
    parser.add_argument("-t", "--interval", type=int, default=300, help="Time interval for fetching server.")
    args, _ = parser.parse_known_args()
    
    while True:
        os.makedirs(args.data_dir, exist_ok=True)
        download_data(data_dir=args.data_dir, interval=args.interval)

main()
