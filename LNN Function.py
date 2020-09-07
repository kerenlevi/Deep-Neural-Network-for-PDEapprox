import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from matplotlib import pyplot as plt
from Neural_Network_Initializer import *
import numpy as np
import pandas as pd
import seaborn as sns


# Helmholtz solution function and it's derivatives
def fun_analytic(a, b, M):
    o = np.multiply(np.cos(a * M[:, 0]), np.sin(b * M[:, 1]))
    return o

def fun_analytic_dx(a, b, M):
    o = np.multiply(- a * np.sin(a * M[:, 0]), np.sin(b * M[:, 1]))
    return o

def fun_analytic_dy(a, b, M):
    o = np.multiply(np.cos(a * M[:, 0]), b * np.cos(b * M[:, 1]))
    return o


# The Gradient given a function U &  X - the variable to deffirntiate by
def MyGrad(U, X):
    gradu = tf.gradients(U, X)
    u_x = gradu[0][:, 0]
    u_x = tf.squeeze(u_x)
    u_x = tf.reshape(u_x, [-1, 1])
    u_y = gradu[0][:, 1]
    u_y = tf.squeeze(u_y)
    u_y = tf.reshape(u_y, [-1, 1])
    return u_x, u_y


# The Laplacian function given given a function U &  X - the variable to deffirntiate by

def MyHess(U, X):
    [ux, uy] = MyGrad(U, X)
    [u_xx, _] = MyGrad(ux, X)
    [_, u_yy] = MyGrad(uy, X)

    return u_xx + u_yy

def LNN_Helmholtz_Approx(PDE_Param ,Neural_Network_Param, Loss_Param, Optimizer_Param):
    """
    Arguments -

        PDE Parameters:
            a, b - Eigenvalue parameters
            Ns - Number of points inside the domain
            Nb - Number of boundary points
            Nrand -
            epsil -
            lowBound, upBound - Domain Upper and Lower Bounds

        Neural Network architecture - This is a 4 Layer Linear neural network with tanh(x) activation in 3 layers,
         the last layer has only liner operation:
            input_num_units - x, y
            hidden_num_units - 3 layers with hidden_num_units neurons
            hidden_num_units_last - last layer size
            output_num_units - u(x,y) - the approximated solution

        Network Optimiser:
            learning_rate  - decay rate for Adam Optimizer
            epochs - Total number of epochs to train the network
            ReduceFactor - decay rate to Gradient descent step every (*) epochs
            lrReduceEvery - Number of
            EarlyStop
    """

    a = PDE_Param["a"]
    b = PDE_Param["b"]
    Ns = PDE_Param["Ns"]
    Nb = PDE_Param["Nb"]
    Nrand = PDE_Param["Nrand"]
    epsil = PDE_Param["epsil"]
    lowBound = PDE_Param["lowBound"]
    upBound = PDE_Param["upBound"]


    # Domain Data Points for the Neural Network

    # Random points inside the domain [lowBound,upBound] * [lowBound,upBound]
    np.random.seed(10)
    rand_points = (upBound - lowBound - 2 * epsil) * np.random.random_sample((Nrand, 2)) + lowBound + epsil

    # Boundary points
    b = np.linspace(lowBound, upBound, Nb)
    Tb_x_0 = np.zeros((Nb, 2), dtype=np.float32)
    Tb_x_0[:, 0] = lowBound
    Tb_x_0[:, 1] = b
    Tb_x_1 = np.zeros((Nb, 2), dtype=np.float32)
    Tb_x_1[:, 0] = upBound
    Tb_x_1[:, 1] = b
    Tb_y_0 = np.zeros((Nb, 2), dtype=np.float32)
    Tb_y_0[:, 0] = b
    Tb_y_0[:, 1] = lowBound
    Tb_y_1 = np.zeros((Nb, 2), dtype=np.float32)
    Tb_y_1[:, 0] = b
    Tb_y_1[:, 1] = upBound

    Tb = np.concatenate((Tb_x_0, Tb_x_1, Tb_y_0, Tb_y_1), axis=0)
    T0 = fun_analytic(a,b, Tb)


    # Neuman boundary Conditions
    Ns1 = fun_analytic_dy(a, b, Tb_y_0)
    Ns2 = fun_analytic_dx(a, b, Tb_x_1)
    Ns3 = fun_analytic_dy(a, b, Tb_y_1)
    Ns4 = fun_analytic_dx(a, b, Tb_x_0)


    # Neural Network Architecture Patameters
    input_num_units = Neural_Network_Param["input_num_units"]
    hidden_num_units = Neural_Network_Param["hidden_num_units"]
    hidden_num_units_last = Neural_Network_Param["hidden_num_units_last"]
    output_num_units = Neural_Network_Param["output_num_units"]

    # Domain, Boundary  & Neumann Condition PlaceHolders
    t1 = tf.placeholder(tf.float32, [None, input_num_units])  # Domain variable placeholder #
    t0 = tf.placeholder(tf.float32, [None, input_num_units])  # Boundary variable placeholder #

    n1 = tf.placeholder(tf.float32, [None, input_num_units])  # Neumann condition variable placeholder #
    n2 = tf.placeholder(tf.float32, [None, input_num_units])
    n3 = tf.placeholder(tf.float32, [None, input_num_units])
    n4 = tf.placeholder(tf.float32, [None, input_num_units])



    NN = Neural_Network3Layers(input_num_units, hidden_num_units, hidden_num_units, hidden_num_units_last,
                               output_num_units)

    T = NN.forward(t1)
    laplacian = MyHess(T, t1)

    N1 = NN.forward(n1)
    [_, NN_1] = MyGrad(N1, n1)
    N2 = NN.forward(n2)
    [NN_2, _] = MyGrad(N2, n2)
    N3 = NN.forward(n3)
    [_, NN_3] = MyGrad(N3, n3)
    N4 = NN.forward(n4)
    [NN_4, _] = MyGrad(N4, n4)



# Loss function -

    miu = Loss_Param["miu"]
    lamb = Loss_Param["lamb"]
    K = Loss_Param["K"]
    alpha = Loss_Param["alpha"]

    topk = tf.nn.top_k(tf.reshape(tf.abs(laplacian + (a**2 + b**2) * T), (-1,)), K)

    loss1 = lamb * tf.reduce_mean(tf.square(laplacian + (a**2 + b**2) * T))

    loss2 = miu * tf.reduce_mean(topk.values)

    loss3 = tf.reduce_mean(tf.abs(NN.forward(t0) - T0))

    loss4 = tf.reduce_mean(tf.abs(NN_1 - Ns1)) + tf.reduce_mean(tf.abs(NN_2 - Ns2)) \
            + tf.reduce_mean(tf.abs(NN_3 - Ns3)) + tf.reduce_mean(tf.abs(NN_4 - Ns4))

    reg_losses = 2 * alpha * (
                tf.nn.l2_loss(NN.W1) + tf.nn.l2_loss(NN.W2) + tf.nn.l2_loss(NN.W3) + tf.nn.l2_loss(NN.W4) +
                tf.nn.l2_loss(NN.b1) + tf.nn.l2_loss(NN.b2) + tf.nn.l2_loss(NN.b3) + tf.nn.l2_loss(NN.b4))


    loss = loss1 + loss2 + loss3 + loss4 + reg_losses


    # Neural Network Optimizer (Minimizer)

    learning_rate = Optimizer_Param["learning_rate"]
    epochs = Optimizer_Param["epochs"]
    ReduceFactor = Optimizer_Param["ReduceFactor"]
    lrReduceEvery = Optimizer_Param["lrReduceEvery"]
    EarlyStop = Optimizer_Param["EarlyStop"]


    optimizer_a = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        min_loss = 1e+20
        counter = 0

        for i in range(1, epochs + 1):
            _, total_loss, l1, l2, l3, l4 = sess.run([optimizer_a, loss, loss1, loss2, loss3, loss4],
                                                     feed_dict={t1: rand_points, t0: Tb, n1: Tb_y_0, n2: Tb_x_1,
                                                                n3: Tb_y_1, n4: Tb_x_0})
            print(
                "Epoch {0}: total: {1:.6e}   l1:{2:.4e}    l2: {3:.4e}  l3:{4:.4e} l4:{5:4e}".format(i, total_loss, l1,
                                                                                                     l2, l3, l4))

            # Early stop in case there is no improvement in the error for an iteration
            if i % lrReduceEvery == 0:
                learning_rate = learning_rate * ReduceFactor
            if total_loss < min_loss:
                counter = 0
                min_loss = total_loss
                print('**')
            else:
                counter += 1
            if counter > EarlyStop:
                print('Early stop {0:.4e}'.format(min_loss))
                break



        # 4 Different graphs - Ground Truth inside omega, and boundary, Neural Network's Solution

        # Domain Data Points for the plot
        t_space = np.zeros((Ns, 2), dtype=np.float32)
        t_space[:, 0] = np.linspace(lowBound + epsil, upBound - epsil, Ns)
        t_space[:, 1] = np.linspace(lowBound + epsil, upBound - epsil, Ns)

        x, y = np.meshgrid(t_space[:, 0], t_space[:, 1])
        positions = np.vstack([x.ravel(), y.ravel()])
        positions = np.transpose(positions)

        fig, axes = plt.subplots(nrows=2, ncols=2)

        ser1 = pd.Series(data=positions[:, 0])
        ser2 = pd.Series(data=positions[:, 1])

        # Omega GT
        Ugt = fun_analytic(positions)

        df_Ugt = ser1.to_frame(name='x')
        df_Ugt['y'] = ser2
        df_Ugt['z'] = pd.Series(data=Ugt)
        pivot_Ugt = pd.pivot_table(df_Ugt, values='z', index='y', columns='x')
        sns.heatmap(pivot_Ugt, ax=axes[0, 1], cmap='magma', xticklabels=False, yticklabels=False)
        axes[0, 1].set_title('Ground Truth - Domain')

        # Omega NN
        Uhat = sess.run([T], feed_dict={t1: positions})  # LB
        Uhat = np.squeeze(Uhat)
        err = np.mean(np.square(Ugt - Uhat))

        df_Uhat = ser1.to_frame(name='x')
        df_Uhat['y'] = ser2
        df_Uhat['z'] = pd.Series(data=Uhat)
        pivot_Uhat = pd.pivot_table(df_Uhat, values='z', index='y', columns='x')
        sns.heatmap(pivot_Uhat, ax=axes[0, 0], cmap='magma', xticklabels=False, yticklabels=False)

        axes[0, 0].set_title('Neural Network - Domain mse={:0.4f}'.format(err))

        print('---------mse=', err)

        # Boundary NN
        serr1 = pd.Series(data=Tb[:, 0])
        serr2 = pd.Series(data=Tb[:, 1])

        Uhat_b = sess.run([T], feed_dict={t1: Tb})
        Uhat_b = np.squeeze(Uhat_b)

        df_Uhat_b = serr1.to_frame(name='x')
        df_Uhat_b['y'] = serr2
        df_Uhat_b['z'] = pd.Series(data=Uhat_b)

        pivot_Uhat_b = pd.pivot_table(df_Uhat_b, values='z', index='y', columns='x')
        sns.heatmap(pivot_Uhat_b, ax=axes[1, 0], cmap='magma', xticklabels=False, yticklabels=False)
        axes[1, 0].set_title('Neural Network - Boundary')

        # Boundary GT
        Ugt_b = fun_analytic(Tb)
        df_Ugt_b = serr1.to_frame(name='x')
        df_Ugt_b['y'] = serr2
        df_Ugt_b['z'] = pd.Series(data=Ugt_b)

        pivot_Ugt_b = pd.pivot_table(df_Ugt_b, values='z', index='y', columns='x')

        sns.heatmap(pivot_Ugt_b, ax=axes[1, 1], cmap='magma', xticklabels=False, yticklabels=False)
        axes[1, 1].set_title('Ground Truth - Boundary')

        plt.tight_layout()
        plt.show()






PDE_Param = {"a": 1,"b": 1,
            "Ns": 300, "Nb" : 300,
            "Nrand": 10*4000, "epsil": 1e-2,
            "lowBound": -np.pi / 4, "upBound":  np.pi / 4,}

Neural_Network_Param = {"input_num_units" : 2,
                        "hidden_num_units" : 10,
                        "hidden_num_units_last" : 10,
                        "output_num_units" : 1}

Loss_Param = {"miu" : float(1e-2), "lamb" : 1,
                "K" : 40, "alpha" : (1e-8)}

Optimizer_Param = {"learning_rate" : 0.01,
                    "epochs" : 1200,
                    "ReduceFactor" : 0.8, "lrReduceEvery" : 200,
                    "EarlyStop" : 20 }

LNN_Helmholtz_Approx(PDE_Param, Neural_Network_Param, Loss_Param, Optimizer_Param)