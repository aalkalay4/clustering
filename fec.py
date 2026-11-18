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


    def merge_small(self, X, current, n):
        post_merge = current.copy()
        ids, counts = np.unique(post_merge, return_counts=True) #finds how many per cluster
        if len(ids) <= 1:                                   #only 1 cluster
            return post_merge
        s_min = max(2, int(n / 20))   #checking if any cluster has less than min
        large = counts >= s_min
        small = counts < s_min
        if not np.any(small):  #if none are small, all good!
            return post_merge
        if not np.any(large):  #if they're all small, there is no next big one, put them all together in biggest?
            big_cluster = ids[np.argmax(counts)]
            return np.full_like(post_merge, big_cluster)  #fill all with big cluster
        #if some small clusters, move them to large ones near
        large_clusters = ids[large]
        small_clusters = ids[small]
        large_centers = []
        for l_cluster in large_clusters:    #find store the center of each large cluster
            points = X[post_merge == l_cluster]
            cluster_center = points.mean(axis=0)
            large_centers.append(cluster_center)
        large_centers = np.array(large_centers)

        for s_cluster in small_clusters:    #move small ones to nearest large one
            s_points = X[post_merge == s_cluster]
            s_cluster_center = s_points.mean(axis=0)
            distance = np.linalg.norm(large_centers - s_cluster_center, axis=1)
            closest_l = large_clusters[np.argmin(distance)]
            post_merge[post_merge == s_cluster] = closest_l

        unique_values, merged = np.unique(post_merge, return_inverse=True)  #grab the values of where to merge
        return merged

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
                w_ij[i,j_num] = np.exp(-pairwise_distances[i,j]/(self.sigma**2))

        #if 1 is neighbors with 2, 2 should be neighbors with 1
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
                edge_weights[i,j] = np.exp(-pairwise_distances[i,j]/(self.sigma**2))



        #2. initialize flux
        f_1 = np.zeros((self.T+1, n))  #to store at each t
        f_1[0,:] = 1                   #initializing all f(0) = 1

        #3. simulate flow
        #downhill criteria
        downhill_check = self.centroid_distances[None,:] <= (self.centroid_distances[:,None] + (.1 * self.sigma))
        downhill_edge_weights = np.where(downhill_check, edge_weights, 0)  #keep values for downhill, else 0

        np.fill_diagonal(downhill_edge_weights, 0)  #fill self with zero too

        #flux aware edge weights
        final_outgoing = np.zeros(n)  #adding in loop for sink check later

        for t in range(self.T):
            f_t = f_1[t]
            phi=FluxEquilibriumClustering.compute_phi(self,t,f_1[t])
            if np.ndim(phi) == 0:  #if phi = 1 (not an array) no change
                flux_edge = downhill_edge_weights
            else:
                flux_edge = phi[:,None] * downhill_edge_weights  #need to flatten phi if it isn't 1 then apply to edge weights

            #summing
            weight_sum = np.sum(flux_edge, axis=1, keepdims=True) # (n,1) to divide later
            weight_sum[weight_sum == 0] = 1 #to catch /0
            #send flux along each edge formula
            flux_out = self.alpha * f_t[:,None] * flux_edge / weight_sum
             #storing last flux out
            if t == self.T - 1:
                final_flux_out = flux_out.copy()

            total_out = np.sum(flux_out, axis=1) #total sent
            final_outgoing = total_out           #only need the last one stored
            total_in = np.sum(flux_out, axis=0)      #total received

            f_1[t + 1] = f_t - total_out + total_in

        #4. detect sinks
        final_flux = f_1[-1]  #flux at final iteration
        sinks_check = (final_outgoing < self. epsilon) & (final_flux > np.mean(final_flux))  #last one (T is out of range because this gets it to T)
        if not np.any(sinks_check):  #if no sinks found, point closest to centroid is now sink
            sinks_check[np.argmin(self.centroid_distances)] = True
        sinks = np.where(sinks_check)
        sinks = sinks[0]  #want to keep the array in the tuple that gives

        #5. assign clusters w cycle detection
        sink_labels = np.full(n, -1)
        for index, is_sink in enumerate(sinks):
            sink_labels[is_sink] = index

        next_steepest = np.full(n, -1)
        for i in range(n):
            if sinks_check[i] == True:  #sink node
                continue
            if not np.any(downhill_edge_weights[i] >0): #dead end
                next_steepest[i] = sinks[argmin(pairwise_distances[i, sinks])]  #go to nearest instead
            else:
                steepest = np.argmax(np.where(downhill_edge_weights[i] > 0, final_flux_out[i], -np.inf))  #flux if downhill, else -inf
                next_steepest[i] = steepest


        #cycle detection
        labels = np.full(n, -1)
        labels[sinks_check] = sink_labels[sinks_check]
        cycle_labels = set()
        for i in range(n):
            if labels[i] != -1:
                continue
            path = []
            current = i
            seen = set()
            cycle_found = False
            while labels[current] == -1:  #not a sink
                if current in seen:   #cycle
                    cycle_found = True
                    break
                seen.add(current)
                path.append(current)
                current = next_steepest[current]

            if cycle_found:  #if hit endless cycle make node a sink
                new_sink = current
                sinks_check[new_sink] = True
                sink_labels[new_sink] = len(sinks)
                sinks = np.append(sinks, new_sink)
                new_label = sink_labels[new_sink]
                cycle_labels.add(new_label)
            else:
                new_label = labels[current]
            for node in path:
                labels[node] = new_label

        #6. post processing: small cluster merging
        for l_cycle in list(cycle_labels):
            cycle_center = X[labels == l_cycle].mean(axis=0)
            best = np.inf #first will always be better
            best_label = None
            for l_other in np.unique(labels):
                if l_other == l_cycle:
                    continue
                other_center = X[labels == l_other].mean(axis=0)
                distance = np.linalg.norm(other_center - cycle_center)
                if distance < best:
                    best = distance
                    best_label = l_other
            labels[labels == l_cycle] = best_label

        labels = self.merge_small(X, labels, n)

        flux = final_flux
        sinks = sinks_check

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
