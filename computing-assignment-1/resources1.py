import numpy

# Load a CSV file
def load_csv(filename, last_column_str=False, normalize=False, as_int=False, filter_data=None):
    dataset, head, classes = tools_rnd.generate_data(filename)
    if dataset is None:
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
        dataset = numpy.array(dataset)
        if not last_column_str and len(numpy.unique(dataset[:,-1])) <= 10:
            classes = dict([("%s" % v, v) for v in numpy.unique(dataset[:,-1])])
    if normalize:
        dataset = normalize_dataset(dataset)
    if as_int:
        dataset = dataset.astype(int)
    if filter_data is not None:
        dataset, head, classes = filter_data(dataset, head, classes)
    return dataset, head, classes

# Find the min and max values for each column
def dataset_minmax(dataset):
    return tools_visu.dataset_minmax(dataset)

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax=None):
    if minmax is None:
        minmax = dataset_minmax(dataset)
    return (dataset - numpy.tile(minmax[0, :], (dataset.shape[0], 1))) / numpy.tile(minmax[1, :]-minmax[0, :], (dataset.shape[0], 1))

# Sample k random points from the domain 
def sample_domain(k, minmax=None, dataset=None):
    if dataset is not None:
        minmax = dataset_minmax(dataset)
    if minmax is None:
        return numpy.random.random(k)
    d = numpy.random.random((k, minmax.shape[1]))
    return numpy.tile(minmax[0, :], (k, 1)) + d*numpy.tile(minmax[1, :]-minmax[0, :], (k, 1))

# Compute distances between two sets of instances
def euclidean_distance(A, B):
    return numpy.vstack([numpy.sqrt(numpy.sum((A - numpy.tile(B[i,:], (A.shape[0], 1)))**2, axis=1)) for i in range(B.shape[0])]).T

def L1_distance(A, B):
    return numpy.vstack([numpy.sum(numpy.abs(A - numpy.tile(B[i,:], (A.shape[0], 1))), axis=1) for i in range(B.shape[0])]).T

# Calculate contingency matrix
def contingency_matrix(actual, predicted, weights=None):
    if weights is None:
        weights = numpy.ones(actual.shape[0], dtype=int)
    ac_int = actual.astype(int)
    prd_int = predicted.astype(int)
    counts = numpy.zeros((numpy.maximum(2,numpy.max(prd_int)+1), numpy.maximum(2,numpy.max(ac_int)+1), 2), dtype=type(weights[0]))
    for p,a,w in zip(prd_int, ac_int, weights):
        counts[p, a, 0] += 1
        counts[p, a, 1] += w
    return counts

# Calculate metrics from confusion matrix
def TPR_CM(confusion_matrix):
    if confusion_matrix[1,1] == 0: return 0.
    return (confusion_matrix[1,1])/float(confusion_matrix[1,1]+confusion_matrix[0,1])
def TNR_CM(confusion_matrix):
    if confusion_matrix[0,0] == 0: return 0.
    return (confusion_matrix[0,0])/float(confusion_matrix[1,0]+confusion_matrix[0,0])
def FPR_CM(confusion_matrix):
    if confusion_matrix[1,0] == 0: return 0.
    return (confusion_matrix[1,0])/float(confusion_matrix[1,0]+confusion_matrix[0,0])
def FNR_CM(confusion_matrix):
    if confusion_matrix[0,1] == 0: return 0.
    return (confusion_matrix[0,1])/float(confusion_matrix[1,1]+confusion_matrix[0,1])
def recall_CM(confusion_matrix):
    return TPR_CM(confusion_matrix)
def precision_CM(confusion_matrix):
    if confusion_matrix[1,1] == 0: return 0.
    return (confusion_matrix[1,1])/float(confusion_matrix[1,1]+confusion_matrix[1,0])
def accuracy_CM(confusion_matrix):
    if (confusion_matrix[0,0]+confusion_matrix[1,1]) == 0: return 0.
    return (confusion_matrix[0,0]+confusion_matrix[1,1])/float(numpy.sum(confusion_matrix))


import cvxopt.solvers
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
    return numpy.dot(X, Y.T)
Kernel.kfunctions['linear'] = linear
def polynomial(X, Y, degrees=None, offs=None):
    if degrees is None:
        return linear(X, Y)
    if offs is None:
        return numpy.sum(numpy.dstack([numpy.dot(X, Y.T)**d for d in degrees]), axis=2)
    return numpy.sum(numpy.dstack([(numpy.dot(X, Y.T)+offs[i])**d for i,d in enumerate(degrees)]), axis=2)
Kernel.kfunctions['polynomial'] = polynomial
def RBF(X, Y, sigma):
    return numpy.vstack([numpy.exp(-numpy.sum((X-numpy.outer(numpy.ones(X.shape[0]), Y[yi,:]))** 2, axis=1) / (2. * sigma ** 2)).T for yi in range(Y.shape[0])]).T
Kernel.kfunctions['RBF'] = RBF

def compute_multipliers(X, y, c, kernel):
    n_samples, n_features = X.shape
    
    K = kernel.distance_matrix(X)
    P = cvxopt.matrix(numpy.outer(y, y) * K)
    q = cvxopt.matrix(-1 * numpy.ones(n_samples))
    G = cvxopt.matrix(numpy.eye(n_samples)*-1)
    h = cvxopt.matrix(numpy.zeros(n_samples))
    # G = cvxopt.matrix(numpy.vstack((numpy.eye(n_samples)*-1, numpy.eye(n_samples))))       
    # h = cvxopt.matrix(numpy.hstack((numpy.zeros(n_samples), numpy.ones(n_samples) * c)))    
    A = cvxopt.matrix(numpy.array([y]), (1, n_samples))
    b = cvxopt.matrix(0.0)
    
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)

    # Lagrange multipliers
    return numpy.ravel(solution['x'])
