import numpy as np
from sklearn.cluster import KMeans

from scipy.stats import norm
import SimpleITK as sitk

from neuromodex_vnet_dbs.models.ProbabilisticBasedSegmentationModel import ProbabilisticBasedSegmentationModel


class GMMSegmentationModel(ProbabilisticBasedSegmentationModel):

    """
    Own implementation of GMM
    """

    def __init__(self, seg_name="GMM", max_iter=150, tol=1e-4, **kwargs):
        super().__init__(seg_name=seg_name, **kwargs)

        self.log_likelihoods = []
        self.resp = None
        self.means = np.zeros([self.n_components, 1])
        self.covariances = np.zeros([self.n_components, 1])
        self.weights = np.zeros([self.n_components])
        self.max_iter = max_iter
        self.tol = tol

    def segment(self, sitk_image: sitk.Image):
        self._log("Starting segmentation")

        image = sitk.GetArrayFromImage(sitk_image)

        image_array = image.astype(np.float32)

        X = image_array[image_array > 0].reshape(-1, 1)
        X = (X - np.mean(X)) / np.std(X)

        N = X.shape[0]

        kmeans = KMeans(n_clusters=self.n_components, n_init='auto')
        kmeans.fit(X)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_.flatten()

        resp = None

        # INIT
        for n in range(self.n_components):
            cluster_data = X[labels == n]

            self.means[n] = centers[n]
            self.covariances[n] = np.var(cluster_data)
            self.weights[n] = len(cluster_data) / N

        # sort components by mean
        sort_indices = np.argsort(self.means[:, 0])
        self.means = self.means[sort_indices]
        self.covariances = self.covariances[sort_indices]
        self.weights = self.weights[sort_indices]

        self.plot_gmm(X)

        for iteration in range(self.max_iter):
            # -- E-Step --
            resp = np.zeros((N, self.n_components))

            for k in range(self.n_components):
                resp[:, k] = self.weights[k] * norm.pdf(X.flatten(), self.means[k], np.sqrt(self.covariances[k]))

            resp_sum = resp.sum(axis=1, keepdims=True)
            resp /= resp_sum

            # -- M-Step --
            for k in range(self.n_components):
                Nk = resp[:, k].sum()
                self.means[k] = (resp[:, k].T @ X) / Nk
                self.covariances[k] = (resp[:, k].T @ (X - self.means[k]) ** 2) / Nk
                self.weights[k] = Nk / N

            # Log-likelihood
            ll = np.sum(np.log(resp_sum))
            self.log_likelihoods.append(ll)

            if self.verbose and iteration % 10 == 0:
                self._log(f"Segmentation Iteration {iteration}: {ll}")
                self.plot_gmm(X, f"GMM for iteration {iteration}")

            # Termination condition
            if iteration > 1 and np.abs(self.log_likelihoods[-1] - self.log_likelihoods[-2]) < self.tol:
                break

        self.plot_gmm(X, f"Final GMM for iteration {iteration}")
        if resp is None:
            self._log(f"No responsibilities found in {self.max_iter} Iterations ")
        else:
            self.labels = resp
