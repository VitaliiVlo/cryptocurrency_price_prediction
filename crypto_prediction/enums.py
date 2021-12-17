from enum import Enum


class InfoType(str, Enum):
    PRICE = 'price'
    TWEETS = 'tweets'
    TRENDS = 'google_trends'
