#### This file contains pure numpy framework of vanila and convolutional neural network
import numpy as np

def xavier(shape):
    """"
    This is the xavier initialization function
    http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf equation (16)
    """
    if len(shape) == 4:
        init_range = np.sqrt(6. / (np.prod(shape[:-1]) + shape[-1]))
    elif len(shape) == 2:
        init_range = np.sqrt(6. / (shape[-2] + shape[-1]))
    else:
        init_range = np.sqrt(6. / shape[-1])
    return np.random.uniform(-init_range, init_range, shape).astype(np.float64)

def softmax(X):
    """""
    Softmax function maps Rˆk  >> [0, 1]ˆK
    """
    normalizer = np.sum(np.exp(X - np.max(X, axis=-1, keepdims=True)),
                        axis=-1, keepdims=True)
    logits = np.exp(X - np.max(X, axis=-1, keepdims=True)) \
             / np.tile(normalizer, (1, X.shape[1]))
    return logits

def accuracy_evaluation(predict, actual):
    total = predict.shape[0]
    result = np.sum(np.argmax(predict, axis=-1) == np.argmax(actual, axis=-1))
    return np.sum(result) / total

class linear(object):
    """"
    Linear layer object
    """
    def __init__(self, name, shape, initializer):
        self.name = name
        self.W = initializer(shape)
        self.b = initializer([shape[1]])
        self.params = [self.W, self.b]
        print('Module :' + self.name + 'is constructed')

    def __call__(self, X):
        return self.forward_pass(X)

    def forward_pass(self, X):
        return X.dot(self.W) \
               + np.tile(self.b, X.shape[0]).reshape((X.shape[0], -1))

    def backward_pass(self, dL_dy):
        return dL_dy.dot(self.W.T)

    def param_gradients(self, dL_dy, X):
        self.dL_dW = X.T.dot(dL_dy)
        self.dL_db = np.sum(dL_dy, axis=0)
        self.grads = [self.dL_dW, self.dL_db]


class relu:
    """"
    Relu layer
    """
    def __init__(self, name):
        self.name = name
        print('Module : ' + self.name + ' is constructed')

    def __call__(self, X):
        return self.forward_pass(X)

    def forward_pass(self, X):
        self.relu_map = (X > 0)
        return X * (X > 0)

    def backward_pass(self, dL_dy):
        return dL_dy * self.relu_map

class flatten:

    def __init__(self, name):
        self.name = name
        print('Module : ' + self.name + ' is constructed')

    def __call__(self, X):
        return self.forward_pass(X)

    def forward_pass(self, X):
        self.original_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward_pass(self, dL_dy):
        return dL_dy.reshape(self.original_shape)


class maxpooling:

    def __init__(self, name, shape, padding, stride):
        # shape = (fh, fw)
        # stride = (sh, sw)
        # X_shape = (batch, h, w, c)
        self.name = name
        self.s = s
        self.shape = shape
        if padding == 'SAME':
            self.p_h = int((shape[0] - 1) / 2)
            self.p_w = int((shape[1] - 1) / 2)
        elif padding == 'VALID':
            self.ph = 0
            self.pw = 0
        print('Module : ' + self.name + ' is constructed')

    def __call__(self, X):
        return self.forward_pass(X)

    def forward_pass(self, X):
        o_h = int(np.floor((X.shape[1] + 2 * self.p_h - self.shape[0]) / self.s[0])) + 1
        o_w = int(np.floor((X.shape[2] + 2 * self.p_w - self.shape[1]) / self.s[1])) + 1
        o = np.zeros((X.shape[0], o_h, o_w, X.shape[-1]))
        X = np.pad(X,
                   [(0, 0),
                    (self.p_h, self.p_h),
                    (self.p_w, self.p_w),
                    (0, 0)],
                   mode='constant',
                   constant_values=0)
        self.maxpool_x = X
        for k in range(o_h):
            for l in range(o_w):
                o[:, k, l, :] = \
                np.max(X[:, self.s[0]*k:self.s[0]*k+self.shape[0],
                         self.s[1]*l:self.s[1]*l+self.shape[1], :],
                       axis=(1,2))
        self.maxpool_o = o
        return o


    def backward_pass(self, dL_dy):
        dL_dX = np.zeros(self.maxpool_x.shape)
        for k in range(dL_dy.shape[1]):
            for l in range(dL_dy.shape[2]):
                dim1_low = k*self.s[0]
                dim1_up = k*self.s[0]+self.shape[0]
                dim2_low = l*self.s[1]
                dim2_up = l*self.s[1]+self.shape[1]
                d = self.maxpool_x[:,
                                   dim1_low:dim1_up,
                                   dim2_low:dim2_up,
                                   :]
                dL_dX[:,
                      dim1_low:dim1_up,
                      dim2_low:dim2_up,
                      :] += (d == self.maxpool_o[:, k:k+1, l:l+1, :]) \
                    * np.tile(dL_dy[:, k:k+1, l:l+1, :],
                              (1, self.shape[0], self.shape[1], 1))
        if self.p_h == 0 and self.p_w == 0:
            dL_dX = dL_dX[:, :, :, :]
        elif self.p_h == 0 and self.p_w != 0:
            dL_dX = dL_dX[:, :, self.p_w:-self.p_w, :]
        elif self.p_h != 0 and self.p_w == 0:
            dL_dX = dL_dX[:, self.p_h:-self.p_h, :, :]
        elif self.p_h != 0 and self.p_w != 0:
            dL_dX = dL_dX[:, self.p_h:-self.p_h, self.p_w,-self.p_w, :]
        return dL_dX


class convolution:
    def __init__(self, name, shape, initializer, padding, stride):
        # shape = (fh, fw, cin, cout)
        # stride = (sh, sw)
        # X_shape = (batch, h, w, c)
        self.name = name
        self.W = initializer(shape)
        self.b = initializer([shape[-1]])
        self.params = [self.W, self.b]
        self.s = stride
        if padding == 'SAME':
            self.p_h = int((shape[0] - 1) / 2)
            self.p_w = int((shape[1] - 1) / 2)
        elif padding == 'VALID':
            self.p_h = 0
            self.p_w = 0
        print('Module : ' + self.name + ' is constructed')

    def __call__(self, X):
        return self.forward_pass(X)

    def forward_pass(self, X):
        o_h = int(np.floor((X.shape[1] + 2 * self.p_h \
                            - self.W.shape[0]) / self.s[0])) + 1
        o_w = int(np.floor((X.shape[2] + 2 * self.p_w \
                            - self.W.shape[1]) / self.s[1])) + 1
        o = np.zeros((X.shape[0], o_h, o_w, self.W.shape[-1]))
        X = np.pad(X,
                   [(0, 0),
                    (self.p_h, self.p_h),
                    (self.p_w, self.p_w),
                    (0,0)],
                   mode='constant',
                   constant_values=0)
        W = np.tile(self.W.reshape((1,)+self.W.shape),
                    (X.shape[0],)+(1, 1, 1, 1))
        X = np.tile(X.reshape(X.shape+(1,)),
                    (1, 1, 1, 1)+(self.W.shape[-1],))
        for k in range(o_h):
            for l in range(o_w):
                c = np.sum(X[:,
                           self.s[0]*k:self.s[0]*k+self.W.shape[0],
                           self.s[1]*l:self.s[1]*l+self.W.shape[1],
                           :, :] * W,
                           axis=(1, 2, 3))
                c = c.reshape(c.shape + (1, 1)).transpose(0, 2, 3, 1)
                o[:, k:k+1, l:l+1, :] = c
        b = np.tile(self.b.reshape((1, 1, 1) + self.b.shape),
                    (X.shape[0], o_h ,o_w, 1))
        o += b
        return o

    def backward_pass(self, dL_dy, X):
        # N x h x w x cin
        X = np.pad(X,
                   [(0, 0), (self.p_h, self.p_h),
                    (self.p_w, self.p_w),
                    (0, 0)],
                   mode='constant',
                   constant_values=0)
        dL_dX = np.zeros(X.shape)
        # N x h x w x cin x cout
        W = np.tile(self.W.reshape((1, )+self.W.shape),
                    (X.shape[0], )+(1, 1, 1,1))
        # N x h x w x cin x cout
        dL_dy = np.tile(dL_dy.reshape(dL_dy.shape+(1, )),
                        (1, 1, 1, 1)+(W.shape[3], ))\
                  .transpose(0, 1, 2, 4, 3)

        for k in range(dL_dy.shape[1]):
            for l in range(dL_dy.shape[2]):
                dL_dX[:,
                      self.s[0]*k:self.W.shape[0]+self.s[0]*k,
                      self.s[1]*l:self.W.shape[1]+self.s[1]*l,
                      :] += np.sum(np.tile(dL_dy[:, k:k+1, l:l+1, :, :],
                                          (1, )+W.shape[1:3]+(1, 1))\
                                   * W[:, :, :, :, :], axis=4)
        if self.p_h == 0 and self.p_w == 0:
            dL_dX = dL_dX[:, :, :, :]
        elif self.p_h == 0 and self.p_w != 0:
            dL_dX = dL_dX[:, :, self.p_w:-self.p_w, :]
        elif self.p_h != 0 and self.p_w == 0:
            dL_dX = dL_dX[:, self.p_h:-self.p_h, :, :]
        elif self.p_h != 0 and self.p_w != 0:
            dL_dX = dL_dX[:, self.p_h:-self.p_h, self.p_w:-self.p_w, :]
        return dL_dX

    def param_gradients(self, dL_dy, X):
        X = np.pad(X,
                   [(0, 0), (self.p_h, self.p_h),
                    (self.p_w, self.p_w), (0, 0)],
                   mode='constant', constant_values=0)
        # N x h x w x cin x cout
        X = np.tile(X.reshape(X.shape + (1, )),
                    (1, 1, 1, 1)+(dL_dy.shape[-1],))
        self.dL_db = np.sum(dL_dy, axis=(0, 1, 2))
        # N x h x w x cin x cout
        dL_dy = np.tile(dL_dy.reshape(dL_dy.shape + (1, )),
                        (1, 1, 1, 1)+(X.shape[-2],))\
                        .transpose(0, 1, 2, 4, 3)
        self.dL_dW = np.zeros(self.W.shape)
        for k in range(dL_dy.shape[1]):
            for l in range(dL_dy.shape[2]):
                self.dL_dW[:, :, :, :] += \
                np.sum(np.tile(dL_dy[:, k:k+1, l:l+1, :, :],
                               (1, )+self.W.shape[:2]+(1, 1))\
                       * X[:,
                           k*self.s[0]:k*self.s[0]+self.W.shape[0],
                           l*self.s[1]:l*self.s[1]+self.W.shape[1],
                           :, :], axis=0)
        self.grads = [self.dL_dW, self.dL_db]

class cross_entropy_softmax_logits:
    def __init__(self, name):
        self.name = name
        print('Module : ' + self.name + ' is constructed')

    def __call__(self, X, Y):
        return self.forward_pass(X, Y)

    def forward_pass(self, X, Y):
        return np.sum(np.sum(-Y * np.log(softmax(X) + 1e-7), axis=-1))

    def backward_pass(self, X, Y):
        return softmax(X) - Y







