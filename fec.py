from __future__ import annotations
from typing import Optional
import numpy as np


class FluxEquilibriumClustering:
    """Flux Equilibrium Clustering (FEC) with improved defaults.

    Parameters
    ----------
    k : int, default=10
        Number of nearest neighbours when building the k-NN graph.
        For better clustering, consider k in range [8, 15].
    sigma : Optional[float], default=None
        Gaussian kernel bandwidth for the initial edge weights. If ``None``,
        infer from data. Recommended: use a fraction of median pairwise distance
        (e.g., 0.2 * median_distance) for more discriminative weights.
    alpha : float, default=0.5
        Fraction of stored flux that can be redistributed at each iteration.
        Higher values (0.5-0.8) promote more aggressive clustering.
    T : int, default=25
        Number of transport iterations. Consider 30-50 for better convergence.
    epsilon : float, default=1e-4
        Threshold on the *outgoing* flux below which a node is considered a sink.
        Lower values (1e-4 to 1e-6) help create fewer, larger clusters.
    beta : Optional[float], default=None
        Controls the non-linear reinforcement when re-weighting edges.

        * ``beta is None``  –  use ``phi(f) = log(1+f)`` (recommended).
        * ``beta = 0``      –  recovers the *original* FEC behaviour.
        * any other value   –  use ``phi(f) = f**beta``.

    Notes
    -----
    Implementation hints for better clustering:
    - Handle small datasets: ensure k <= n-1 to avoid boundary errors
    - Consider relaxed downhill flow: allow flux to similar-distance neighbors
    - Add cycle detection in path-following to prevent infinite loops
    - Post-process: merge very small clusters with nearest larger ones
    """

    def __init__(
        self,
        k: int = 10,
        sigma: Optional[float] = None,
        alpha: float = 0.5,
        T: int = 25,
        epsilon: float = 1e-4,
        beta: Optional[float] = None,
    ) -> None:
        # Hint: self.k = k, self.sigma = sigma, etc.
        self.k = k
        self.sigma = sigma
        self.alpha = alpha
        self.T = T
        self.epsilon = epsilon
        self.beta = beta

        self.global_centroid = None
        self.centroid_distances = None
        self.labels = None
        #raise NotImplementedError

    #to get phi(f) for flux aware edge weights
    def compute_phi(self, t, f):
        if t == 0:
            return 1
        if self.beta == 0:
            return 1
        if self.beta is None:
            return np.log(1+f)
        if self.beta != 0:
            return f**self.beta

    def fit(self, X: np.ndarray) -> "FluxEquilibriumClustering":
        """Compute cluster labels for *X*.

        Implementation steps:
        1. Compute global centroid and distances to centroid
        2. Build symmetric k-NN graph with Gaussian edge weights
           - Handle small datasets: use effective_k = min(k, n-1)
           - If sigma is None, infer from data (e.g., fraction of median distance)
        3. Initialize flux: f_i = 1 for all nodes
        4. Run T flux-transport iterations:
           - Apply flux-aware reweighting (beta parameter)
           - Use relaxed downhill criterion (optional improvement)
           - Update flux: f_i(t+1) = f_i(t) - outgoing + incoming
        5. Identify sinks: nodes with outgoing flux < epsilon
        6. Assign clusters via steepest downhill paths
           - Add cycle detection to prevent infinite loops
           - Consider post-processing to merge small clusters

        The final labels must be stored in ``self.labels``.
        """
        # Hint: Start with computing centroid = X.mean(axis=0)

        # Add your code here
        n = X.shape[0]
        if n == 0:
            raise ValueError("n cannot be zero")

        #1. pre-compute
        self.global_centroid = X.mean(axis=0)
        self.centroid_distances = np.linalg.norm(X - self.global_centroid, axis=1)

        subtracted = X[:,None,:] - X[None,:,:]                  #stretching in both directions to broadcast then xi-xj
        pairwise_distances = np.sum(subtracted**2, axis=-1)     #axis -1 to sum over last dimension

        if self.sigma is None:  #then = 0.2 * median pairwise distance if i < j
            upper_i, upper_j = np.triu_indices(n=n, k=1, m=None) #Return the indices for the upper-triangle of array.Triu is array with the elements below the k-th diagonal zeroed. Want above 0 diag so axis 1.
            median_ilessj_dist = np.median(np.sqrt(pairwise_distances[upper_i, upper_j]) )
            self.sigma = 0.2 * median_ilessj_dist

        keff = min(self.k, n - 1)

        #finding only keff nearest neighbors
        distance_neighbors = np.argsort(pairwise_distances, axis =1)
        kNN_neighbors = distance_neighbors[:, 1:keff+1]  #start at 1 or else first point will be closest, got cut at keff, need to make it inclusive of it
        #storing w_ij
        w_ij = np.zeros((n, keff)) #only want k nearest for each
        for i in range (n):
            for j_num, j in enumerate(kNN_neighbors[i]):
                w_ij[i,j_num] = np.exp(-pairwise_distances[i,j] / (self.sigma**2))

        #if 1 is neighbors with 2, 2 is neighbors with 1
        kNN_neighbors_all = []
        for i in range(n):
            kNN_neighbors_all.append(set(kNN_neighbors[i]))
        for i in range(n):
            for j in kNN_neighbors[i]:
                kNN_neighbors_all[j].add(i)

        # edge weights for all of those
        edge_weights = np.zeros((n,n))
        for i in range (n):
            for j in kNN_neighbors_all[i]:
                edge_weights[i,j] = np.exp(-pairwise_distances[i,j] / (self.sigma**2))

        #2. initialize flux
        f_1 = np.zeros((self.T+1, n))  #to store at each t
        f_1[0] = 1                   #initializing all f(0) = 1

        #3. simulate flow
        #flux aware edge weights
        change = []
        for t in range(self.T):
            phi=FluxEquilibriumClustering.compute_phi(self,t,f_1[t])
            if np.ndim(phi) == 0:  #if phi = 1 (not an array) make all that value
                phi = np.full(n, phi)

            change_t = np.zeros((n, keff))
            outgoing = np.zeros(n)
            incoming = np.zeros(n)
            for i in range(n):
                neighbors = [j for j in kNN_neighbors[i]
                             if self.centroid_distances[j] <= self.centroid_distances[i] + (.1 * self.sigma)]  #downhill check
                if len(neighbors) == 0:
                    continue #sink
                flux_weight = phi[neighbors] * edge_weights[i, neighbors]              #adjusted flux_weight of neighbors with phi
                changes = self.alpha * flux_weight * f_1[t][i] / np.sum(flux_weight)   #given formula of sending flux
                change_t[i, :len(changes)] = changes
                outgoing[i] = np.sum(changes)
                for j,c in zip(neighbors,changes):
                    incoming[j] += c
            change.append(change_t)
            f_1[t+1] = f_1[t] - outgoing + incoming  #flux at each after iteration

        #4. detect sinks
        sinks_check = outgoing < self. epsilon  #checks for sinks with final outgoing flux at each node
        if not np.any(sinks_check):  #if no sinks found, point closest to centroid is now sink
            sinks_check[np.argmin(self.centroid_distances)] = True
        sinks = np.where(sinks_check)  #actual locations of the sinks
        sinks = sinks[0]  #want to keep the array in the tuple from np.where

        #5. assign clusters w cycle detection
        sink_l = np.full(n, -1)   #sink cluster label l
        for index, is_sink in enumerate(sinks):  #each sink gets unique cluster label
            sink_l[is_sink] = index

        steepest = np.full(n, -1)  # to hold each steepest neighbor for non sinks
        for i in range(n):
            if sink_l[i] != -1:  # sink
                continue
            # find neighbors again to check steepest
            neighbors = [j for j in kNN_neighbors[i]
                         if self.centroid_distances[j] <= self.centroid_distances[i] + (.1 * self.sigma)]  # downhill check
            steepest_loc = np.argmax(change[-1][i, :len(neighbors)])    #see which of nearest neighbor is steepest in terms of change in flux
            steepest[i] = neighbors[steepest_loc]  #storing which neighbor it is
        sink_count = len(sinks)
        for i in range(n):
            if sink_l[i] != -1:
                continue
            path = []
            current = i
            seen = {}  #to store nodes/positions
            while True:
                if sink_l[current] != -1:  #path reached a sink
                    for each_step in path:
                        sink_l[each_step] = sink_l[current]  #to assign all nodes in the same path the cluster label as sink
                    break  #do not stay in path after sink
                if current in seen:   #already hit node in path --> cycle
                    sink_l[current] = sink_count  #add next sink cluster label
                    sink_count += 1
                    for each_step in path:
                        sink_l[each_step] = sink_l[current] #label all steps in path the same sink node
                    break
                seen[current] = len(path)
                path.append(current)   #keep track of current path
                current = steepest[current]

        #6. post processing: small cluster merging
        while True:
            cluster, counts = np.unique(sink_l, return_counts=True)  # finds how many per cluster
            if len(cluster) <= 1:  # only 1 cluster
                break
            s_min = max(2, int(n / 20))  # getting min to see if clusters need to be merges
            small = cluster[counts < s_min]  #grouping small v large clusters found
            large = cluster[counts >= s_min]
            if len(small) == 0:  #no small clusters
                break

            large_centers = []
            for l_cluster in large:  # find store the center of each large cluster
                points = X[sink_l == l_cluster]       #points in the cluster
                cluster_center = points.mean(axis=0)  #centroid
                large_centers.append(cluster_center)
            large_centers = np.array(large_centers)

            for s_cluster in small:  # move small ones to nearest large one
                s_points = X[sink_l == s_cluster]         #points in the cluster
                s_cluster_center = s_points.mean(axis=0)  #centroid
                distance = np.linalg.norm(large_centers - s_cluster_center, axis=1)
                closest_l = large[np.argmin(distance)]    #nearest large centroid
                sink_l[sink_l == s_cluster] = closest_l   #merging the small one into the near large one

        flux = f_1[-1]
        sinks = sinks_check
        labels = sink_l

        # ---- DO NOT CHANGE THE REST OF THIS METHOD ---- #
        # We require you maintain these attributes for testing purpose.
        self.w_ij = w_ij
        self.flux = flux
        self.sinks = sinks
        self.labels = labels

        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Convenience method that returns the cluster labels."""
        self.fit(X)
        return self.labels

    def __repr__(self) -> str:
        params = (
            f"k={getattr(self, 'k', '?')}",
            f"sigma={getattr(self, 'sigma', '?')}",
            f"alpha={getattr(self, 'alpha', '?')}",
            f"T={getattr(self, 'T', '?')}",
            f"epsilon={getattr(self, 'epsilon', '?')}",
            f"beta={getattr(self, 'beta', '?')}",
        )
        return f"{self.__class__.__name__}({', '.join(params)})"
