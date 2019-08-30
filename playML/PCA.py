import numpy as np


class PCA:

    def __init__(self, n_components):
        """初始化PCA"""
        assert n_components >= 1, "n_components must be valid"
        self.n_components = n_components
        self.components_ = None

    def fit(self, x, eta=0.01, n_iters=1e4):
        """获取数据集X的前n个主成分"""

        def demean(X):
            return X - np.mean(X, axis=0)

        def f(w, X):
            return np.sum((X.dot(w) ** 2)) / len(X)

        def df(w, X):
            return X.T.dot(X.dot(w)) * 2 / len(X)

        def direction(w):
            return w / np.linalg.norm(w)

        def first_components(X, initial_w, eta, n_iters=1e4, epsilon=1e-8):

            w = direction(initial_w)
            i_iter = 0

            while i_iter < n_iters:

                gradient = df(w, X)
                last_w = w
                w = w + eta * gradient
                w = direction(w)  # 每次求一个单位方向

                if abs(f(w, X) - f(last_w, X)) < epsilon:
                    break

                i_iter += 1
            return w

        X_pca = demean(x)
        self.components_ = np.empty(shape=(self.n_components, x.shape[1]))

        for i in range(self.n_components):
            initial_w = np.random.random(X_pca.shape[1])
            w = first_components(X_pca, initial_w, eta, n_iters=1e4)
            self.components_[i, :] = w

            X_pca = X_pca - X_pca.dot(w).reshape(-1, 1) * w

            return self

    def transform(self,X):
        """将给定的X，映射到个个主成分分量中"""

        return X.dot(self.components_.T)

    def inverse_transform(self,X):
        """将给定的X，反向映射回原来的特征空间"""
        return X.dot(self.components_)

    def __repr__(self):
        return "PCA(n_components = %d" % self.n_components
