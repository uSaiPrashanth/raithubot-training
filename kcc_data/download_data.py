import requests
import json
from tqdm import trange
import argparse
from threading import Thread
from queue import Queue

def parse_data(district, state, year, queue):
    all_data = []
    for month in trange(1, 13):
        payload['StateCD'] = state
        payload['DistrictCd'] = district
        payload['Year'] = year
        payload['Month'] = month

        res = requests.get(url, payload, timeout=100)
        try:
            if res.json()['ResponseCode'] == '1':
                all_data.extend(res.json()['data'])
        except:
            pass 
    
    queue.put(all_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type = int)
    parser.add_argument("--district", type = int)
    parser.add_argument("--store-path", type = str)

    args = parser.parse_args()
    url = "https://dackkms.gov.in/Account/API/kKMS_QueryData.aspx"

    payload = {
        'StateCD': '01',
        'DistrictCd': '0104',
    }
    all_data = []
    state = args.state
    district = args.district
    
    all_data = Queue()
    for year in trange(2008, 2024):
        t = Thread(target = parse_data, args = (district, state, year, all_data))
        t.start()
    
    results = []
    for i in range(16):
        data = all_data.get()
        results.extend(data)
    
    all_data.task_done()
    
    with open(f'{args.store_path}/data_{district}_{state}.json', 'w') as f:
        json.dump(results, f)