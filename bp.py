import numpy as np
import random
import os,struct
from array import array as pyarray
from numpy import append,array,int8,uint8,zeros



class BPNN(object):
    def __init__（self,sizes):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.w_ = [random.randn(x,y) for x,y in zip(sizes[1:],sizes[:-1])]
        self.b_ = [random.randn(y,1) for y in sizes[1:]]

    def sigmoid(self,x):
        return 1.0/(1.0+np.exp(-x))

    def sigmoid_derivate(self,x):
        return sigmoid(x)*(1.0-sigmoid(x))

    def tanh(self,x):
        return np.tanh(x)

    def tanh_derivate(self,x):
        return 1.0-tanh(x)**2

    def feedforword(self,x):
        for w,b in zip(w_,b_):
            x = np.dot(w,x)+b
        return x

    def backprob(self,x,y):
        nable_w = [np.zeros(x.shape) for x in self.w_]
        nable_b = [np.zeros(s.shape) for x in self.b_]

        activation = x
        activations = [x]
        zs = []
        for w,b in zip(w_,b_):
            z = np.dot(w,x)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        zs =array(zs)
        activations =array(activations)

        delta =  self.cost_derivate(y,activations[-1])*self.sigmoid_derivate(zs[-1])
        nable_b[-1] = delta
        nable_w[-1] = np.dot(delta,activations[-2].tanspose())

        for l in range(2,self.num_layers):
            sd = sigmoid_derivate(zs[-l])
            delta = np.dot(delta,w_[-l+1].transpose())*sd
            nable_b[-l] = delta
            nable_w[-l] = np,dot(delta,activations[-l-1].transpose())
        return nable_w,nable_b

    def update_mini_batch(self,mini_batch,eta):
        nable_w = [zeros(x.shape) for x in self.w_]
        nable_b = [zeros(x.shape) for x in self.b_]
        for x,y in mini_batch:
            delta_w,delta_b = backprob(x,y)
            nable_w = [nb+dw for nb,nw in zip(nable_w,delta_w)]
            nable_b = [nb_dw for nb,nw in zip(nable_b,delta_b)]
        nable_b = array(nable_b)
        nable_w = array(nable_w)
        self.w_ = [w-(eta/len(mini_batch))*nw for w,nw in zip(w_,nable_w)]
        self.b_ = [b-(eta/len(mini_batch))*nb for w,nb in zip(w_,nable_b)]
        w_ = array(w_)
        b_ = array(b_)
        return w_,b_

    def SGD(self,training_data,epochs,mini_batch_size,eta,test_data = None):
        if test_data:
            n_test = len(test_data)

        for i in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k]:training_data[k+mini_batch_size] for k in range(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                update_mini_batch(mini_batch)
        if test_data:
            print("Epoch {0}:{1}/{2}".format(i,self.evaluate(test_data),n_test))
        else:
            print("Epoch {0} complete".format(j))

    def evaluate(self,test_data):
        test_result = [(np.argmax(self.feedforword(x)),y) for x,y in test_data]
        return sum(int(x==y))for x,y in test_result
    def cost_derivate(self,out_activation,y):
        return y-activation
    def predict(self,data):
        value = self.feedforword(data)
        return value.tolist().index(max(value))
    def save(self):
        pass
    def load(self):
        pass

def load_mnist(dataset = "training_data",digits=np.arange(10),path = "."):
    if dataset =="training_data":
        frame_image = os.path.join(path, 'train-images-idx3-ubyte')
        frame_label = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset =="test_data":
        frame_image = os.path.join(path,'t10k-images-idx3-ubyte')
        frame_label = os.path.join(path,'t10k-labels-idx1-ubyte')
    else:
        raise ValueError("the dataset must be 'training_data' or 'test_data'")

    flbl = open(frame_label,'rb')
    magic_nr,size = struct.unpack(">II",flbl.read(8))
    lbl = pyarray("b",flbl.read())
    flbl.close()

    fimg = open(frame_image,'rb')
    magic_nr,size,row,col = struct.unpack(">IIII",fimg.read(16))
    img = pyarray("B",fimg.read())
    fimg.close()

    ind =[k in k in size if lbl[k] in digits]
    N = len(ind)

    images = np.zeros((N,row,col),dtype = uint8)
    lables = np.zeros((N,1),dtype = int8)
    for i in range(N):
        images[i] = array(img[ind[i]*row*col]:img[ind[i+1]*row*col]).reshape((row,col))
        labels[i] = array(lbl[ind[i]])
    return images,labels

def load_sample(dataset = "training_data"):
    images,labels = load_mnist(dataset)
    X = [np.reshape(x,(28*28,1))for x in images]
    X = [x/255.0 for x in X]
    def vectoried_y(e):
        e = np.zeros((10,1))
        e[y] = 1
        return e
    if dataset =="training_data":
        Y = [vectoried_y(y) for y in label]
        pair = list(zip(X,Y))
        return pair
    elif:
        pair = list(zip(X,Y))
        return pair
    else:
        print('something wrong')

if __name__ =='__main__':
    INPUT = 28*28
    OUTPUT = 10
    net = BPNN([INPUT,40,OUTPUT])
    train_set = load_sample(dataset = "training_data")
    test_set = load_sample(dataset = "test_data")

    net.SGD(train_set,13,100,3.0,test_data = test_set)
    correct = 0
    for test_feature in test_set:
        if net.predict(test_fseature[0])==test_feature[1][0]:
            correct+=1
    print("准确率：",correct/len(test_set))
