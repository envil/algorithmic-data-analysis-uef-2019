import cvxopt
import numpy as np
import csv
import matplotlib.pyplot as plt

MAX_INT = np.iinfo(np.int32)
MIN_SUPPORT_VECTOR_MULTIPLIER = 1e-5
cvxopt.solvers.options['show_progress'] = False


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


def compute_multipliers(X, y, kernel, c=None):
    n_samples, n_features = X.shape

    K = kernel.distance_matrix(X)
    P = cvxopt.matrix(np.outer(y, y) * K)
    q = cvxopt.matrix(-1 * np.ones(n_samples))
    if c == None:
        G = cvxopt.matrix(np.eye(n_samples) * -1)
        h = cvxopt.matrix(np.zeros(n_samples))
    else:
        G = cvxopt.matrix(np.vstack((np.eye(n_samples) * -1, np.eye(n_samples))))
        h = cvxopt.matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * c)))
    A = cvxopt.matrix(np.array([y]), (1, n_samples))
    b = cvxopt.matrix(0.0)

    solution = cvxopt.solvers.qp(P, q, G, h, A, b)
    # Lagrange multipliers
    return np.ravel(solution['x']), K


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


def svm_hard(x, y, kernel):
    alphas, K = compute_multipliers(X, y, kernel)
    alphas[alphas < MIN_SUPPORT_VECTOR_MULTIPLIER] = 0
    i = np.argmax(alphas)
    w = (alphas * y).T @ x
    b = y[i] - np.dot(x[i], w)
    return w, b, np.array(alphas > MIN_SUPPORT_VECTOR_MULTIPLIER)


def svm_soft(x, y, c, kernel):
    alphas, K = compute_multipliers(X, y, kernel, c=c)
    alphas[alphas < MIN_SUPPORT_VECTOR_MULTIPLIER] = 0
    w = (alphas * y).T @ x
    # calculation of b was taken from https://pythonprogramming.net/soft-margin-kernel-cvxopt-svm-machine-learning-tutorial/
    b = 0
    a = alphas[alphas > 0]
    sv = alphas > 0
    sv_y = y[sv]
    ind = np.where(alphas > 0)[0]
    for i in range(a.shape[0]):
        b += sv_y[i]
        b -= np.sum(a * sv_y * K[ind[i], sv])
    b /= a.shape[0]

    return w, b, np.array(alphas > MIN_SUPPORT_VECTOR_MULTIPLIER)


def predict(x, w, b):
    return np.sign((np.sum(np.array([w * x_i for x_i in x]), axis=1) + b))


def plot_separating_line(w, b, bounds):
    x = np.linspace(bounds[0], bounds[1], 1000)
    y = (w[0] * x + b) / -w[1]
    plt.plot(x, y, '-r')


# Load data
irisSV, h, c = load_csv('iris-SV-sepal.csv', last_column_str=True)
irisVV, h, c = load_csv('iris-VV-length.csv', last_column_str=True)
creditDE, h, c = load_csv('creditDE.csv')

# =========================== Task 1 ===========================
X, Y = irisSV[:, 0:2], irisSV[:, -1]
Y[Y == 0] -= 1
polynomial_kernel = Kernel('linear')
W, b, support_vectors = svm_hard(X, Y, polynomial_kernel)
Y_predicted = predict(X, W, b)
plt.scatter(X[:, 0], X[:, 1], c=Y_predicted)
plt.title('Hard-margin SVM with IrisSV data set')
plot_separating_line(W, b, [np.min(X[:, 0]), np.max(X[:, 0])])
plt.scatter(X[:, 0][support_vectors], X[:, 1][support_vectors], c='red', s=10)
plt.show()
print('Hard-margin SVM with IrisSV data set equation:')
print('w = {}'.format(W))
print('b = {}'.format(b))

C = 2
X, Y = irisVV[:, 0:2], irisVV[:, -1]
Y[Y == 0] -= 1
polynomial_kernel = Kernel('polynomial')

W, b, support_vectors = svm_soft(X, Y, C, polynomial_kernel)
Y_predicted = predict(X, W, b)
plt.scatter(X[:, 0], X[:, 1], c=Y_predicted)
plt.title('Soft-margin SVM with IrisVV data set')
plot_separating_line(W, b, [np.min(X[:, 0]), np.max(X[:, 0])])
plt.scatter(X[:, 0][support_vectors], X[:, 1][support_vectors], c='red', s=10)
plt.show()
print('w = {}'.format(W))
print('b = {}'.format(b))



X, Y = creditDE[:, 0:-2], creditDE[:, -1]
Y[Y == 0] -= 1
polynomial_kernel = Kernel('RBF')
W, b, support_vectors = svm_hard(X, Y, polynomial_kernel)
Y_predicted = predict(X, W, b)
# plt.scatter(X[:, 0], X[:, 1], c=Y_predicted)
# plt.title('Hard-margin SVM with IrisSV data set')
# plot_separating_line(W, b, [np.min(X[:, 0]), np.max(X[:, 0])])
# plt.scatter(X[:, 0][support_vectors], X[:, 1][support_vectors], c='red', s=10)
# plt.show()
print('Hard-margin SVM with Credit data set equation:')
print('w = {}'.format(W))
print('b = {}'.format(b))