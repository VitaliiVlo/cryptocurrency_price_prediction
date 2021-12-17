import os
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import layers
from keras import models
from keras.utils.vis_utils import plot_model
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

warnings.filterwarnings('ignore')

# seed
seed_value = 128
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)


def plot_comparison(df, compare_val):
    fig, ax1 = plt.subplots(figsize=(13, 7), dpi=200)
    color = 'tab:red'
    ax1.set_xlabel('time')
    ax1.set_ylabel('price', color=color)
    ax1.plot(df['date'], df['price'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel(compare_val, color=color)  # we already handled the x-label with ax1
    ax2.plot(df['date'], df[compare_val], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


def plot_history(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


def plot_cm(y_test, y_pred):
    # confidence = 0.5
    # confident_pred = list()
    # for pred in y_pred:
    #     confident_label = 2 if pred.max() < confidence else pred.argmax()
    #     confident_pred.append(confident_label)
    # cm = confusion_matrix(y_test, confident_pred)

    cm = confusion_matrix(y_test, y_pred.argmax(axis=1))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()


def calc_price_diff(df):
    price_differences = list()
    for index, row in df.iterrows():
        if index == (len(df) - 1):
            continue
        next_row = df.iloc[index + 1]
        price_differences.append(abs(next_row['price'] - row['price']))
    return pd.Series(price_differences)


def preprocess_data(df, prev_days, risk_value):
    new_df = list()
    today_data = None
    for index, row in df.iterrows():
        if index < prev_days:
            continue

        is_today = index == (len(df) - 1)  # today (last) row
        next_row = row if is_today else df.iloc[index + 1]

        if abs(next_row['price'] - row['price']) < risk_value:
            up_down = 2
        else:
            up_down = int(next_row['price'] >= row['price'])

        new_row = [up_down]

        for prev_row_ind in range(prev_days):
            prev_row_ind += 1
            prev_row = df.iloc[index - prev_row_ind]
            tweets_diff = row['tweets'] - prev_row['tweets']
            price_diff = row['price'] - prev_row['price']
            trend_diff = row['trend'] - prev_row['trend']
            new_row.extend([tweets_diff, price_diff, trend_diff])

        if is_today:
            today_data = np.array(new_row[1:]).reshape((1, -1))
            continue
        new_df.append(new_row)

    new_cols = ['up_down']
    for prev_row_ind in range(prev_days):
        prev_row_ind += 1
        new_cols.extend([f'tweets_diff_prev_{prev_row_ind}',
                         f'price_diff_prev_{prev_row_ind}',
                         f'trend_diff_prev_{prev_row_ind}'])

    result = pd.DataFrame(new_df, columns=new_cols)
    return result, today_data


def get_model(X_train):
    model = models.Sequential()

    model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(layers.Dropout(0.2))
    # model.add(layers.BatchNormalization())

    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(3, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model(tweets_path, price_path, trend_path, prev_days, risk_percent):
    crypto_tweets = pd.read_csv(tweets_path, dtype='object')
    crypto_price = pd.read_csv(price_path, dtype='object')
    crypto_trends = pd.read_csv(trend_path, dtype='object')

    df = crypto_tweets.merge(crypto_price, on='date')
    df = df.merge(crypto_trends, on='date')

    df = df[(df.tweets != 'null') & (df.trend != 'null') & (df.price != 'null')]
    df['date'] = pd.to_datetime(df['date'])
    df['price'] = pd.to_numeric(df['price'])
    df['tweets'] = pd.to_numeric(df['tweets'])
    df['trend'] = pd.to_numeric(df['trend'])
    df.reset_index(drop=True, inplace=True)

    # print(pearsonr(df['tweets'], df['price']))
    # print(pearsonr(df['trend'], df['price']))
    # plot_comparison(df, 'tweets')
    # plot_comparison(df, 'trend')

    price_diff = calc_price_diff(df)
    # print(price_diff.quantile([.25, .4, .5, .55, .6, .65, .7, .75, .8, .9]))

    risk_value = price_diff.quantile(risk_percent)
    # print('RV = ', risk_value)
    df, today_data = preprocess_data(df, prev_days=prev_days, risk_value=risk_value)

    # print(df.head())
    # for s in df:
    #     print(df[s].dtype)
    # print(df.shape)
    # print(df['up_down'].tail(50))

    Y = df.iloc[:, 0]
    X = np.array(df.iloc[:, 1:], dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=seed_value)

    # normalization
    std_scale = StandardScaler().fit(X_train)
    X_train = std_scale.transform(X_train)
    X_test = std_scale.transform(X_test)

    model = get_model(X_train)
    history = model.fit(X_train, y_train,
                        epochs=20, batch_size=64,
                        validation_split=0.2,
                        verbose=True)

    test_loss, test_acc = model.evaluate(X_test, y_test)

    print('\n\n')
    # print('K = ', prev_days)
    # print('M = ', risk_percent)
    print('ACC = ', test_acc)
    print('LOS = ', test_loss)
    # print(df['up_down'].value_counts(normalize=True))
    print('Train size = ', X_train.shape)
    print('Test size = ', X_test.shape)

    y_pred = model.predict(X_test)

    # plot_cm(y_test, y_pred)
    # plot_history(history)
    # plot_model(model, to_file='model.png', dpi=200, show_shapes=True, show_layer_names=True)
    # model.save('model.h5')

    today_data = std_scale.transform(today_data)
    tomorrow_prediction = model.predict(today_data).argmax(axis=1)
    if tomorrow_prediction == 0:
        return 'DOWN'
    if tomorrow_prediction == 1:
        return 'UP'
    if tomorrow_prediction == 2:
        return 'RISK'
