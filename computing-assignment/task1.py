import cvxopt
import numpy as np
import csv
import matplotlib.pyplot as plt

MAX_INT = np.iinfo(np.int32)


def load_csv(filename, last_column_str=False, normalize=False, as_int=False, filter_data=None):
    dataset = list()
    head = None
    classes = {}
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for ri, row in enumerate(csv_reader):
            if not row:
                continue
            if ri == 0:
                head = row
            else:
                rr = [r.strip() for r in row]
                if last_column_str:
                    if rr[-1] not in classes:
                        classes[rr[-1]] = len(classes)
                    rr[-1] = classes[rr[-1]]
                dataset.append([float(r) for r in rr])
    dataset = np.array(dataset)
    if not last_column_str and len(np.unique(dataset[:, -1])) <= 10:
        classes = dict([("%s" % v, v) for v in np.unique(dataset[:, -1])])
    # if normalize:
    # dataset = normalize_dataset(dataset)
    if as_int:
        dataset = dataset.astype(int)
    if filter_data is not None:
        dataset, head, classes = filter_data(dataset, head, classes)
    return dataset, head, classes


def compute_multipliers(X, y, kernel, is_soft=False, c=MAX_INT):
    n_samples, n_features = X.shape

    K = kernel.distance_matrix(X)
    P = cvxopt.matrix(np.outer(y, y) * K)
    q = cvxopt.matrix(-1 * np.ones(n_samples))
    if is_soft:
        G = cvxopt.matrix(np.vstack((np.eye(n_samples) * -1, np.eye(n_samples))))
        h = cvxopt.matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * c)))
    else:
        G = cvxopt.matrix(np.eye(n_samples) * -1)
        h = cvxopt.matrix(np.zeros(n_samples))
    A = cvxopt.matrix(np.array([y]), (1, n_samples))
    b = cvxopt.matrix(0.0)

    solution = cvxopt.solvers.qp(P, q, G, h, A, b)

    # Lagrange multipliers
    return np.ravel(solution['x'])


MIN_SUPPORT_VECTOR_MULTIPLIER = 1e-5


class Kernel(object):
    kfunctions = {}
    kfeatures = {}

    def __init__(self, ktype='linear', kparams={}):
        self.ktype = 'linear'
        if ktype in self.kfunctions:
            self.ktype = ktype
        else:
            raise Warning("Kernel %s not implemented!" % self.ktype)
        self.kparams = kparams

    def distance_matrix(self, X, Y=None):
        if Y is None:
            Y = X
        return self.kfunctions[self.ktype](X, Y, **self.kparams)


def linear(X, Y):
    return np.dot(X, Y.T)


Kernel.kfunctions['linear'] = linear


def polynomial(X, Y, degrees=None, offs=None):
    if degrees is None:
        return linear(X, Y)
    if offs is None:
        return np.sum(np.dstack([np.dot(X, Y.T) ** d for d in degrees]), axis=2)
    return np.sum(np.dstack([(np.dot(X, Y.T) + offs[i]) ** d for i, d in enumerate(degrees)]), axis=2)


Kernel.kfunctions['polynomial'] = polynomial


def RBF(X, Y, sigma):
    return np.vstack(
        [np.exp(-np.sum((X - np.outer(np.ones(X.shape[0]), Y[yi, :])) ** 2, axis=1) / (2. * sigma ** 2)).T
         for yi in range(Y.shape[0])]).T


Kernel.kfunctions['RBF'] = RBF

irisSV, h, c = load_csv('iris-SV-sepal.csv', last_column_str=True)
irisVV, h, c = load_csv('iris-VV-length.csv', last_column_str=True)
creditDE, h, c = load_csv('creditDE.csv')


def svm(x, y, kernel, is_soft=False, c=10):
    alphas = compute_multipliers(X, y, kernel, is_soft=is_soft, c=c)
    alphas[alphas < MIN_SUPPORT_VECTOR_MULTIPLIER] = 0
    i = np.argmax(alphas)
    w = (alphas * y) @ x
    b = y[i] - np.dot(w.T, x[i])
    return w, b


def predict(x, w, b):
    y = (np.sum(np.array([w * x_i for x_i in x]), axis=1) + b) >= 1
    return y


X, Y = irisSV[:, 0:2], irisSV[:, -1]
Y[Y == 0] -= 1
linear_kernel = Kernel()
W, b = svm(X, Y, linear_kernel, is_soft=True, c=10)
Y_predicted = predict(X, W, b)
plt.scatter(X[:, 0], X[:, 1], c=Y_predicted)
plt.show()
