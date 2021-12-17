import os
from datetime import date

import pandas as pd
import requests
from bs4 import BeautifulSoup

from crypto_prediction.enums import InfoType


def get_info(data_path, currency, info, extra_info=None):
    data = list()

    session = requests.Session()
    info_to_parse = extra_info or info
    page = session.get(f'https://bitinfocharts.com/comparison/{currency}-{info_to_parse}.html')
    if not page.ok:
        raise ValueError('404!!!')
    soup = BeautifulSoup(page.content, 'html.parser')

    values = str(soup.find_all('script')[4])
    values = values.split('d = new Dygraph(document.getElementById("container"),')[1].split(', {labels: ')[0]
    for i in range(values.count('new Date')):
        date_val = values.split('new Date("')[i + 1].split('"')[0]
        value = values.split('"),')[i + 1].split(']')[0]
        data.append([date_val, value])

    if len(data) < 100:
        print(f'\nNo {currency}_{info} data found, attempt to change!\n')
        if not extra_info:
            return get_info(data_path, currency, info, extra_info='transactions')

    col_name = 'trend' if info == InfoType.TRENDS else info
    result = pd.DataFrame(data, columns=['date', col_name])

    today = date.today()
    result.to_csv(os.path.join(data_path, f'{currency}_{info}_{today}.csv'),
                  index=False, header=True)
