import pickle
import os
import time
import datetime
import random


def generate_pickle(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(path, encoding="latin1"):
    with open(path, 'rb') as handle:
        data = pickle.load(handle, encoding=encoding)
    return data

def timeStamp2time(timeStamp):
    timeArray = time.localtime(timeStamp)
    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    return otherStyleTime

def time2timeStamp(timeS):
    standard_time = "2022-10-25 14:09:33"
    timeS = timeS[:len(standard_time)]
    s_t = time.strptime(timeS, "%Y-%m-%d %H:%M:%S")
    mkt = int(time.mktime(s_t))
    return mkt

def current_timestamp():
    return int(time.time())

def zero_today():
    now = datetime.datetime.now()
    zero_today = now - datetime.timedelta(hours=now.hour, minutes=now.minute, seconds=now.second, microseconds=now.microsecond)
    return zero_today

def zero_yesterday():
    return zero_today() +datetime.timedelta(days=-1)

def shuffle_dict(source_dict):
    l = list(source_dict.items())
    random.shuffle(l)
    d = dict(l)
    return d



if __name__ == '__main__':


    this_time = "2022-09-01 00:00:00"
    #
    # now_time = datetime.datetime.now()
    #
    # print(str(now_time))
    # print(time2timeStamp(str(now_time)))
    print(time2timeStamp(this_time))
    # # print(timeStamp2time(time2timeStamp(this_time)))
    #
    # print(timeStamp2time(time.time()))

    # print(zero_today())
    # print(zero_yesterday())

    # s = {'a':1, 'b':2, 'c':3, 'd':4}
    # print(shuffle_dict(s))
    # print(shuffle_dict(s))
    # print(shuffle_dict(s))

    pass
