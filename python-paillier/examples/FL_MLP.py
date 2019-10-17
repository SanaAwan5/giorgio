"""
This example involves learning using sensitive medical data from multiple hospitals
to predict diabetes progression in patients. The data is a standard dataset from
sklearn[1].

Recorded variables are:
- age,
- gender,
- body mass index,
- average blood pressure,
- and six blood serum measurements.

The target variable is a quantitative measure of the disease progression.
Since this measure is continuous, we solve the problem using linear regression.

The patients' data is split between 3 hospitals, all sharing the same features
but different entities. We refer to this scenario as horizontally partitioned.

The objective is to make use of the whole (virtual) training set to improve
upon the model that can be trained locally at each hospital.

50 patients will be kept as a test set and not used for training.

An additional agent is the 'server' who facilitates the information exchange
among the hospitals under the following privacy constraints:

1) The individual patient's record at each hospital cannot leave the premises,
   not even in encrypted form.
2) Information derived (read: gradients) from any hospital's dataset
   cannot be shared, unless it is first encrypted.
3) None of the parties (hospitals AND server) should be able to infer WHERE
   (in which hospital) a patient in the training set has been treated.

Note that we do not protect from inferring IF a particular patient's data
has been used during learning. Differential privacy could be used on top of
our protocol for addressing the problem. For simplicity, we do not discuss
it in this example.

In this example linear regression is solved by gradient descent. The server
creates a paillier public/private keypair and does not share the private key.
The hospital clients are given the public key. The protocol works as follows.
Until convergence: hospital 1 computes its gradient, encrypts it and sends it
to hospital 2; hospital 2 computes its gradient, encrypts and sums it to
hospital 1's; hospital 3 does the same and passes the overall sum to the
server. The server obtains the gradient of the whole (virtual) training set;
decrypts it and sends the gradient back - in the clear - to every client.
The clients then update their respective local models.

From the learning viewpoint, notice that we are NOT assuming that each
hospital sees an unbiased sample from the same patients' distribution:
hospitals could be geographically very distant or serve a diverse population.
We simulate this condition by sampling patients NOT uniformly at random,
but in a biased fashion.
The test set is instead an unbiased sample from the overall distribution.

From the security viewpoint, we consider all parties to be "honest but curious".
Even by seeing the aggregated gradient in the clear, no participant can pinpoint
where patients' data originated. This is true if this RING protocol is run by
at least 3 clients, which prevents reconstruction of each others' gradients
by simple difference.

This example was inspired by Google's work on secure protocols for federated
learning[2].

[1]: http://scikit-learn.org/stable/datasets/index.html#diabetes-dataset
[2]: https://research.googleblog.com/2017/04/federated-learning-collaborative.html

Dependencies: numpy, sklearn
"""

import numpy as np
#from sklearn.datasets import load_diabetes
#from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml

import phe as paillier

seed = 43
np.random.seed(seed)


def get_data(n_clients):
    """
    Import the dataset via sklearn, shuffle and split train/test.
    Return training, target lists for `n_clients` and a holdout test set
    """
    print("Loading data")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    X = X / 255.
    y=y.astype(np.float)
    # Add constant to emulate intercept
    X = np.c_[X, np.ones(X.shape[0])]

    # The features are already preprocessed
    # Shuffle
    perm = np.random.permutation(X.shape[0])
    X, y = X[perm, :], y[perm]
    #X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    #X = X / 255.

    # Select test at random
    test_size = 50
    valid_size=5
    test_idx = np.random.choice(X.shape[0], size=test_size, replace=False)
    valid_idx = np.random.choice(X.shape[0], size=valid_size, replace=False)
    train_idx = np.ones(X.shape[0], dtype=bool)
    train_idx[test_idx] = False
    train_idx[valid_idx]=False
    X_valid,y_valid=X[valid_idx,:],y[valid_idx]
    X_test, y_test = X[test_idx, :], y[test_idx]
    X_train, y_train = X[train_idx, :], y[train_idx]

    # Split train among multiple clients.
    # The selection is not at random. We simulate the fact that each client
    # sees a potentially very different sample of patients.
    X, y = [], []
    step = int(X_train.shape[0] / n_clients)
    for c in range(n_clients):
        X.append(X_train[step * c: step * (c + 1), :])
        y.append(y_train[step * c: step * (c + 1)])

    return X, y, X_test, y_test , X_valid, y_valid


def mean_square_error(y_pred, y):
    """ 1/m * \sum_{i=1..m} (y_pred_i - y_i)^2 """
    return np.mean((y - y_pred) ** 2)


def encrypt_vector(public_key, x):
    return [public_key.encrypt(i) for i in x]


def decrypt_vector(private_key, x):
    return np.array([private_key.decrypt(i) for i in x])


def sum_encrypted_vectors(x, y):
    if len(x) != len(y):
        raise ValueError('Encrypted vectors must have the same size')
    return [x[i] + y[i] for i in range(len(x))]

def sum_encrypted_vectors(x, y):
    if len(x) != len(y):
        raise ValueError('Encrypted vectors must have the same size')
    return [x[i] + y[i] for i in range(len(x))]


class Alice:
    """Private key holder. Decrypts the average gradient"""

    def __init__(self):
         #keypair = paillier.generate_paillier_keypair(n_length=key_length)
         #self.pubkey, self.privkey = keypair
         self.model=MLPClassifier(hidden_layer_sizes=(785,), max_iter=10, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)

         self.weights = None
         self.intercept = None

    def generate_paillier_keypair(self, n_length):
        self.pubkey, self.privkey = \
            paillier.generate_paillier_keypair(n_length=n_length)

    def decrypt_aggregate(self, input_model, n_clients):
        return decrypt_vector(self.privkey, input_model) / n_clients

    def fit(self, X, y):
        self.model = self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    """def decrypt_aggr(self, encrypted_scores):
        return [self.privkey.decrypt(s) for s in encrypted_scores]"""


class Bob:
    """Runs linear regression with local data or by gradient steps,
    where gradient can be passed in.

    Using public key can encrypt locally computed gradients.
    """

    def __init__(self, name, X, y, model ,pubkey):
        self.name = name
        self.X, self.y = X, y
        self.weights = np.zeros(X.shape[1])
        self.model = model
        self.pubkey = pubkey
    
    def set_weights(self, weights, intercept):
        self.weights = weights
        self.intercept = intercept

    """def encrypted_score(self, x):
        #Compute the score of `x` by multiplying with the encrypted model,
        #which is a vector of paillier.EncryptedNumber
        score = self.intercept
        _, idx = x.nonzero()
        for i in idx:
            score += x[0, i] * self.weights[i]
        return score"""

    """def encrypted_evaluate(self, X):
        return [self.encrypted_score(X[i, :]) for i in range(X.shape[0])]"""

    def encrypt_gradient(self, gradient):
        """Compute and encrypt gradient.
        When `sum_to` is given, sum the encrypted gradient to it, assumed
        to be another vector of the same size
        """
        #gradient = self.compute_gradient()
        encrypted_gradient = encrypt_vector(self.pubkey, gradient)

        #if sum_to is not None:
         #   return sum_encrypted_vectors(sum_to, encrypted_gradient)
       # else:
        return encrypted_gradient


    def fit(self, n_iter, eta=0.0001):
        #Linear regression for n_iter
        for _ in range(n_iter):
            gradient = self.compute_gradient()
            self.gradient_step(gradient, eta)
        return gradient

    def gradient_step(self, gradient, eta=0.01):
        """Update the model with the given gradient"""
        self.weights -= eta * gradient

    def compute_gradient(self):
        """Compute the gradient of the current model using the training set"""
        y_hat = self.predict(self.X) 
        delta = y_hat - self.y
        return delta.dot(self.X) / len(self.X)

    def predict(self, X):
        #Score test data
        #print(type(X.dot(self.weights)))
        return X.dot(self.weights)

    
    def encrypt_weights(self):
        coef = self.model.coefs_
        encrypted_weights = [self.pubkey.encrypt(coefs_[i])
                             for i in range(coefs_.shape[0])]
        encrypted_intercept = self.pubkey.encrypt(self.model.intercept_[0])
        return encrypted_weights


def federated_learning(X, y, X_test, y_test, X_valid, y_valid, config):
    n_clients = config['n_clients']
    n_iter = config['n_iter']
    names = ['Hospital {}'.format(i) for i in range(1, n_clients + 1)]

    # Instantiate the server and generate private and public keys
    # NOTE: using smaller keys sizes wouldn't be cryptographically safe
    #server = Alice(key_length=config['key_length'])
    alice = Alice()
    # NOTE: using smaller keys sizes wouldn't be cryptographically safe
    alice.generate_paillier_keypair(n_length=1024)
    # Instantiate the clients.
    # Each client gets the public key at creation and its own local dataset
    clients = []
    for i in range(n_clients):
        clients.append(Bob(names[i], X[i], y[i], alice.model, alice.pubkey))

    # The federated learning with gradient descent
    print('Running distributed gradient aggregation for {:d} iterations'
          .format(n_iter))
    print("Fitting...")
    alice.model.fit(X_valid,y_valid)
    for i in range(n_iter):

        # Compute gradients, encrypt and aggregate
        for c in clients:
            gradient=c.fit(n_iter,config['eta'])
            encrypted_grads=np.array([c.encrypt_gradient(gradient) for c in clients])
        #print ("encrypted_aggr.shape", encrypted_aggr.shape)
        en_aggr = encrypted_grads[0]
        for i in range (1,len(encrypted_grads)):
            en_aggr+=sum_encrypted_vectors(en_aggr,encrypted_grads[i])  

        # Send aggregate to server and decrypt it
        aggr = alice.decrypt_aggregate(en_aggr, n_clients)
        
        delta_aggr = np.multiply(aggr , 0.0001)
        clf = alice.model
        for i in range(len(clf.coefs_)):
            number_neurons_in_layer = clf.coefs_[i].shape[1]
            print("num neurons in layer %s %s,", i,number_neurons_in_layer)
            for j in range(number_neurons_in_layer):
                weights = clf.coefs_[i][:,j]
                weights -= delta_aggr
        print(weights.shape)

        # Take gradient steps
    for c in clients:
            c.gradient_step(aggr, config['eta'])

    print('Error (MSE) that Alice gets after running the protocol on test set:')
    #for c in clients:
    y_pred = alice.predict(X_test)
    mse = mean_square_error(y_pred, y_test)
    print('{:s}:\t{:.2f}'.format(c.name, mse))


def local_learning(X, y, X_test, y_test, config):
    n_clients = config['n_clients']
    names = ['Hospital {}'.format(i) for i in range(1, n_clients + 1)]

    # Instantiate the clients.
    # Each client gets the public key at creation and its own local dataset
    alice = Alice()
    # NOTE: using smaller keys sizes wouldn't be cryptographically safe
    alice.generate_paillier_keypair(n_length=1024)
    clients = []
    for i in range(n_clients):
        clients.append(Bob(names[i], X[i], y[i], alice.model, alice.pubkey))

    # Each client trains a linear regressor on its own data
    print('Error (MSE) that each client gets on test set by '
          'training only on own local data:')
    for c in clients:
        #c.set_weights(alice.weights, alice.intercept)
        c.fit(config['n_iter'],config['eta'])
        y_pred = c.predict(X_test)
        mse = mean_square_error(y_pred, y_test)
        print('{:s}:\t{:.2f}'.format(c.name, mse))


if __name__ == '__main__':
    config = {
        'n_clients': 5,
        'key_length': 1024,
        'n_iter': 200,
        'eta': 0.0001,
    }
    X, y, X_test, y_test, X_valid, y_valid = get_data(n_clients=config['n_clients'])
    # first each hospital learns a model on its respective dataset for comparison.
    local_learning(X, y, X_test, y_test, config)
    # and now the full glory of federated learning
    federated_learning(X, y, X_test, y_test, X_valid, y_valid, config)
