import collections
import math
import os
import random
import zipfile
import numpy as np
import urllib
import tensorflow as tf

url = 'http://mattmahoney.net/dc/'

def maybe_download(filename,expected_bytes):
    if not os.path.exists(filename):
        filename,_ = urllib.request.urlretrieve(url+filename,filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified',filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify'+filename+'. Can you get to it with a browser?')
    return filename
filename = maybe_download('text8.zip',31344016)

def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(namelist()[0])).split()
    return data

words = read_data(filename)
print('Data size',len(words))

vocabulary_size = 50000
def build_dataset(word):
    count = [['UNK'],-1]
    count.extend(collections.Counter(words).most_common(vocabulary_size-1))
    dictionary = dict()
    for word,_ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    conut[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(),dictionary.keys()))
    return data,count,dictionary,reverse_dictionary
data,count,dictionary,reverse_dictionary = build_dataset(words)

del words
print('Most common words (+UNK)',count[:5])
print('Simple data',data[:10],[reverse_dictionary[i] for i in data[:10]])

data_index = 0

def generate_batch(batch_size,num_skips,skip_window):
    global data_index
    assert batch_size%num_skips == 0
    assert num_skips <=2*skip_window
    batch = np.ndarray(shape = (batch_size),dtype = np.int32)
    labels = np.ndarray(shape = (batch_size,1),dtype = np.int32)
    span = 2 * skip_window + 1
    Buffer = collections.deque(maxlen = span)

    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index+1)%len(data)
    for i in range (batch_size // num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0,span-1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer [skip_window]
            labels[i * num_skips + j,0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1)%len(data)
    return batch,labels
