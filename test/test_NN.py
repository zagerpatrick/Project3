import scripts.io as io
import scripts.util as util
import scripts.NN as nn
import numpy as np


def test_encoder():
    
    # Setup
    dna = ['ATGCTT']
    res = np.array([[1.],
                    [0.],
                    [0.],
                    [0.],
                    [0.],
                    [1.],
                    [0.],
                    [0.],
                    [0.],
                    [0.],
                    [1.],
                    [0.],
                    [0.],
                    [0.],
                    [0.],
                    [1.],
                    [0.],
                    [1.],
                    [0.],
                    [0.],
                    [0.],
                    [1.],
                    [0.],
                    [0.]])

    # Exercise
    encode = util.one_hot_dna(dna, 6)

    # Verify
    np.testing.assert_array_equal(encode[0], res)


def autoencode():
    
    # Setup
    n_train = 1000
    n_test = 1000

    dim = [8, 3, 8]

    x = util.gen_label_array((dim[0], n_train))

    n_epochs = 2000
    learning_rate = 2e-3

    net =nn.NeuralNetwork(dim, 'logistic')
    
    # Exercise
    net.fit(x, x, n_epochs, learning_rate)

    # Verify
    self.assertTrue(net.loss_list[-1] < net.loss_list[1])
