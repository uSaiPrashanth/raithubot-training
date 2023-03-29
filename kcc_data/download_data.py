import requests
import json
from tqdm import trange
import argparse

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
    
    for year in trange(2008, 2024):
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
                print(res.raw)
    
    with open(f'{args.store_path}/data_{district}_{state}.json', 'w') as f:
        json.dump(all_data, f)