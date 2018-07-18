import matplotlib.pyplot as plt
import sklearn as skl
import time

from gradients import *


class CEC(skl.base.BaseEstimator):
    def __init__(self, n_clusters=3, n_init=100, deletion_threshold=0.05, pdf=pdf_multivariate_split_gaussian):
        self.X = None
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.deletion_threshold = deletion_threshold
        self.pdf = pdf
        self.params = []

    def get_pdf_params(self, k):
        return 0,

    def _init_params(self, clusters):
        pass

    def _init_clusters(self):
        d = np.full(self.X.shape[0], np.inf)
        self.y = np.zeros(self.X.shape[0])
        clusters = [self.X[np.random.randint(0, self.n_clusters)]]
        for k in range(self.n_clusters - 1):
            for i in range(d.shape[0]):
                x = self.X[i]
                for c in clusters:
                    distance = np.linalg.norm(c - x)
                    if distance < d[i]:
                        d[i] = distance
            clusters.append(self.X[np.random.choice(self.X.shape[0], 1, p=d ** 2 / np.sum(d ** 2))][0])

        self.y = np.full(self.X.shape[0], -1)
        for j in range(self.y.shape[0]):
            distance = np.inf
            for i, c in enumerate(clusters):
                new_distance = np.linalg.norm(self.X[j] - c)
                if new_distance < distance:
                    distance = new_distance
                    self.y[j] = i
        self.p = np.full(self.n_clusters, 1 / self.n_clusters)

        self._init_params(clusters)

    def calculate_cluster(self, c):
        pass

    def draw_result(self, iteration=0):
        plt.figure(figsize=(5, 5))
        cols = ['red', 'green', 'blue', 'violet', 'teal', 'magenta', 'sienna', 'orange', 'pink', 'black']
        # levels = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
        for k in range(self.n_clusters):
            if np.sum(self.y == k) > 0:
                x, y = np.mgrid[-0.1:1.1:.005, -0.1:1.1:.005]
                z = np.zeros(x.shape)
                for i in range(x.shape[0]):
                    for j in range(x.shape[1]):
                        z[i, j] = self.pdf(np.array([x[i, j], y[i, j]]), *self.get_pdf_params(k))
                try:
                    cs = plt.contour(x, y, z, colors=cols[k], alpha=1., interpolation='bilinear',
                                     antialiasing=True, linewidths=1)
                    # plt.clabel(cs, inline=1, fontsize=6)
                except:
                    print(z.min(), z.max())
        plt.scatter(self.X[:, 0], self.X[:, 1], c=[cols[i] for i in self.y], alpha=0.2, s=0.6)
        # ========= More info ===========
        # for k in range(self.n_clusters):
        #     if np.sum(self.y == k) > 0:
        #         plt.text(self.cluster_centers_[k, 0], self.cluster_centers_[k, 1], str(k), horizontalalignment='center',
        #                  bbox=dict(facecolor='white', alpha=0.5))
        # ===============================
        plt.savefig("plots/sgcec-%s.png" % iteration)

    def fit(self, X, y=None, clear=False):
        self.X = X
        self._init_clusters()
        self.draw_result(-1)

        for i in range(self.n_init):
            y_copy = self.y.copy()
            for j in range(self.X.shape[0]):
                min_i = 0
                min_v = np.inf
                for k in range(self.n_clusters):
                    if sum(self.y == k) >= self.deletion_threshold * self.y.shape[0]:
                        f = self.pdf(self.X[j], *self.get_pdf_params(k))
                        if f < np.finfo(float).eps:
                            continue
                        v = -np.log(self.p[k]) - np.log(f)
                        if v < min_v:
                            min_v = v
                            min_i = k
                # if self.y[j] != min_i:
                #     print("Przepięcie %s do %s przy energii %s" % (self.y[j], min_i, min_v))
                #     print("Energia w starym klastrze: %s (c=%s)" % (-np.log(self.p[self.y[j]]) - np.log(self.pdf(self.X[j], *self.get_pdf_params(self.y[j]))), self.c[self.y[j]]))
                self.y[j] = min_i
            for c in range(self.n_clusters):
                if self.X[self.y == c, :].shape[0] != 0:
                    self.p[c] = sum(self.y == c) / self.X.shape[0]
                    self.calculate_cluster(c)

            self.draw_result(i)
            print(self.get_bic())

            if np.array_equal(self.y, y_copy):
                break

        return self

    def get_bic(self):
        n = self.X.shape[0]
        k = sum(p.size for p in self.params)
        l = 0
        for j in range(n):
            k = self.y[j]
            # f = max(self.pdf(self.X[j], *self.get_pdf_params(k)), np.finfo(float).eps)
            f = self.pdf(self.X[j], *self.get_pdf_params(k))
            v = np.log(self.p[k]) + np.log(f)
            l += v
        return np.log(n) * k - 2 * l

    def save_model(self):
        pass

    def load_model(self):
        pass


class SGCEC(CEC):
    def __init__(self, n_clusters=3, n_init=100, deletion_threshold=0.05):
        CEC.__init__(self, n_clusters, n_init, deletion_threshold, pdf_multivariate_split_gaussian)

    def get_pdf_params(self, k):
        return self.cluster_centers_[k], self.W[k], self.sigma[k], self.tau[k]

    def calculate_cluster(self, c):
        mu0, W0, _ = optimize_split_gaussian(self.X[self.y == c, :], self.cluster_centers_[c], self.W[c])
        self.cluster_centers_[c] = mu0
        self.W[c] = W0
        self.sigma[c], self.tau[c] = get_sigma_tau(self.X[self.y == c, :], mu0, W0)

    def _init_params(self, clusters):
        self.cluster_centers_ = np.vstack(clusters)
        self.W = np.zeros((self.n_clusters, self.X.shape[1], self.X.shape[1]))
        self.sigma = np.zeros((self.n_clusters, self.X.shape[1]))
        self.tau = np.zeros((self.n_clusters, self.X.shape[1]))
        self.params = [self.W, self.cluster_centers_]
        for k in range(self.n_clusters):
            if self.X[self.y == k, :].shape[0] != 0:
                self.cluster_centers_[k] = self.X[self.y == k, :].mean(axis=0)
                self.W[k] = np.cov(self.X[self.y == k, :].T)
                self.calculate_cluster(k)
        return self.cluster_centers_


class GammaCEC(CEC):
    def __init__(self, n_clusters=3, n_init=100, deletion_threshold=0.05):
        CEC.__init__(self, n_clusters, n_init, deletion_threshold, pdf_multivariate_split_gamma)

    def get_pdf_params(self, k):
        return self.cluster_centers_[k], self.W[k], self.sigma[k], self.tau[k], self.c[k]

    def calculate_cluster(self, c):
        # Experiment
        l = 0.9
        self.c[c] = l * self.c[c] + (1 - l) * 2
        self.cluster_centers_[c] = l * self.cluster_centers_[c] + (1 - l) * self.X[self.y == c, :].mean(axis=0)
        # End
        mu0, W0, c0, _ = optimize_split_gamma(self.X[self.y == c, :], self.cluster_centers_[c], self.W[c], self.c[c])
        self.cluster_centers_[c] = mu0
        self.W[c] = W0
        self.c[c] = c0
        self.sigma[c], self.tau[c], _ = get_gamma_sigmas(self.X[self.y == c, :], mu0, W0, c0)

    def _init_params(self, clusters):
        self.cluster_centers_ = np.vstack(clusters)
        self.W = np.zeros((self.n_clusters, self.X.shape[1], self.X.shape[1]))
        self.sigma = np.zeros((self.n_clusters, self.X.shape[1]))
        self.tau = np.zeros((self.n_clusters, self.X.shape[1]))
        self.c = np.full(self.n_clusters, 2.)
        self.params = [self.W, self.cluster_centers_, self.c]
        for k in range(self.n_clusters):
            if self.X[self.y == k, :].shape[0] != 0:
                self.cluster_centers_[k] = self.X[self.y == k, :].mean(axis=0)
                self.W[k] = np.cov(self.X[self.y == k, :].T)
                self.calculate_cluster(k)
        return self.cluster_centers_
