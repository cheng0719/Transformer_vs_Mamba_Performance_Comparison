import requests, json, datetime, csv
import stockInfo

get_url = stockInfo.url.format(stockInfo.code, stockInfo.start_date, stockInfo.end_date)

result = requests.get(get_url).json()
if(result['result'] == 'success'):
    for item in result['data']:
        timestamp = item["date"]
        date = datetime.datetime.utcfromtimestamp(timestamp)
        item["date"] = date.strftime("%Y%m%d")
    # print(json.dumps(result['data'], indent=4))
    
    json_data = result['data']

    # sort by date
    sorted_json_data = sorted(json_data, key=lambda x: datetime.datetime.strptime(x["date"], "%Y%m%d"))

    # specify fields
    fields = [field for field in sorted_json_data[0].keys() if field != 'turnover' and field != 'capacity' and field != 'change' and field != 'transaction_volume' and field != 'stock_code_id']

    # specify csv file
    csv_file = "./output.csv"

    # write to csv
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for data in sorted_json_data:
            writer.writerow({field: data[field] for field in fields})
    
    print("Data saved to output.csv")
else:
    print('Failed to fetch data from server.')