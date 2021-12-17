import os
from datetime import date, timedelta

from constants import DATA_PATH, CURRENCY, PREV_DAYS, RISK_PERCENT
from crypto_prediction.enums import InfoType
from crypto_prediction.model import train_model
from crypto_prediction.parser import get_info


def main():
    today = date.today()
    for info_type in InfoType:
        check_path = os.path.join(DATA_PATH, f'{CURRENCY}_{info_type}_{today}.csv')
        if not os.path.exists(check_path):
            get_info(DATA_PATH, CURRENCY, info_type)

    tweets_path = os.path.join(DATA_PATH, f'{CURRENCY}_{InfoType.TWEETS}_{today}.csv')
    price_path = os.path.join(DATA_PATH, f'{CURRENCY}_{InfoType.PRICE}_{today}.csv')
    trend_path = os.path.join(DATA_PATH, f'{CURRENCY}_{InfoType.TRENDS}_{today}.csv')

    today_pred = train_model(tweets_path, price_path, trend_path, PREV_DAYS, RISK_PERCENT)
    print(f'Prediction for {CURRENCY} is', today_pred)
    print('Process finished')


if __name__ == '__main__':
    main()
