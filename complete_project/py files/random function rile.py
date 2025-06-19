def calculate_embedding_shift_sub(self, sigma_corr: float = 0.05) -> np.ndarray:
    """Use the transition probability to project the velocity direction on the embedding

    Arguments
    ---------
    sigma_corr: float, default=0.05
        the kernel scaling

    Returns
    -------
    Nothing but it creates the following attributes:
    transition_prob: np.ndarray
        the transition probability calculated using the exponential kernel on the correlation coefficient
    delta_embedding: np.ndarray
        The resulting vector
    """
    # Kernel evaluation
    # NOTE maybe sparse matrix here are slower than dense
    self.transition_prob = np.exp(self.corrcoef / sigma_corr) * self.embedding_knn.toarray()  # naive
    np.save("transition_prob.npy", self.transition_prob)

    print("transition prob: ", self.transition_prob)
    self.transition_prob /= self.transition_prob.sum(1)[:, None]
    print(" normed transition prob: ", self.transition_prob)

    # as for the first row (that we care about in this function, is always only 1's for self.ebmedding_knn, we set all entries in the last column to a 1
    # to avoid that a division by zero occurs, (as no 1's present in self.embedding.knn for a row (small but possible chance)
    # self.transition_prob[:,-1] = 1

    if hasattr(self, "corrcoef_random"):
        logging.debug("Calculate transition probability for negative control")
        self.transition_prob_random = np.exp(self.corrcoef_random / sigma_corr) * self.embedding_knn.toarray()  # naive
        self.transition_prob_random /= self.transition_prob_random.sum(1)[:, None]
    # TODO check if unitary vector is an option as calculated here? is the distance (difference) between the two points calculated correctly still
    # in a 50d space instead of a 2d space
    unitary_vectors = self.embedding.T[:, None, :] - self.embedding.T[:, :, None]  # shape (2,ncells,ncells)
    with np.errstate(divide='ignore', invalid='ignore'):
        unitary_vectors /= np.linalg.norm(unitary_vectors, ord=2, axis=0)  # divide by L2
        for i in range(unitary_vectors.shape[0]):  # Iterate through all PCA components (dimension 0)
            np.fill_diagonal(unitary_vectors[i, ...], 0)  # Zero out diagonal for each component
    self.temp_un = unitary_vectors

    self.delta_embedding = (self.transition_prob * unitary_vectors).sum(2)
    self.delta_embedding -= (self.embedding_knn.toarray() * unitary_vectors).sum(2) / self.embedding_knn.sum(1).A.T
    self.delta_embedding = self.delta_embedding.T

    if hasattr(self, "corrcoef_random"):
        self.delta_embedding_random = (self.transition_prob_random * unitary_vectors).sum(2)
        self.delta_embedding_random -= (self.embedding_knn.toarray() * unitary_vectors).sum(2) / self.embedding_knn.sum(
            1).A.T
        self.delta_embedding_random = self.delta_embedding_random.T

    return self.delta_embedding











batch_size, num_instances_per_batch = batch_idxs.shape
feature_dim = self.adata.n_vars

# Fill with selected instances for each batch
max_idx = np.max([np.max(batch_idxs) for batch_idxs in batch_idxs])
batch_idx_maps = np.full((batch_size, max_idx + 1), -1, dtype=int)
simulation_input = np.zeros((batch_size, num_instances_per_batch, feature_dim))
cluster_labels_present_in_all_batch = set()
for batch_idx, data_idxs in enumerate(batch_idxs):
    simulation_input[batch_idx] = self.adata.X[data_idxs].toarray() if sp.issparse(self.adata.X) else self.adata.X[data_idxs]
    cluster_labels_present_in_all_batch.update(self.adata.obs[self.cluster_column_name][data_idxs])
    batch_idx_maps[batch_idx, data_idxs] = np.arange(len(data_idxs))

gem_imputed = simulation_input.copy()

if batch_size != len(perturb_condition):
    raise ValueError("Batch size and perturb condition size do not match.")

# Apply perturbations
for batch_idx, (gene, value) in enumerate(perturb_condition.items()):
    if gene not in self.gene_to_index_dict:
        print(f"Gene {gene} is not in the subset. Skipping perturbation.")
        continue
    index_of_gene = self.gene_to_index_dict[gene]
    simulation_input[batch_idx, :, index_of_gene] = value  # set perturbation on entire subset
    if not knockout_prev_used:
        continue
    already_perturbed_value = self.prev_perturbed_value_batch_dict[batch_idx]
    for gene, value in already_perturbed_value.items():
        if gene not in self.gene_to_index_dict:
            print(f"Gene {gene} is not in the subset. Skipping perturbation.")
            continue
        simulation_input[batch_idx, :, self.gene_to_index_dict[gene]] = value

if knockout_prev_used:
    for batch_idx, (gene, value) in enumerate(perturb_condition.items()):
        # Update perturbation history
        self.prev_perturbed_value_batch_dict[batch_idx][gene] = value



# Process by cluster
simulated_data = np.zeros_like(simulation_input)
for cluster_label in cluster_labels_present_in_all_batch:
    indices_for_cluster_label = self.cluster_label_to_idx_dict[cluster_label]
    coef_matrix = self.coef_matrix_per_cluster_np_dict[cluster_label]

    simulated_data_batch = []
    original_data_batch = []
    original_indices_map = []
    batch_counts = []

    for batch_idx, subset_idxs in enumerate(batch_idxs):
        cluster_indices_in_batch = np.intersect1d(subset_idxs, indices_for_cluster_label, assume_unique=True)
        if len(cluster_indices_in_batch) == 0:
            continue

        tensor_indices_label = batch_idx_maps[batch_idx][cluster_indices_in_batch]
        simulated_data_label = simulation_input[batch_idx, tensor_indices_label]
        original_data_label = gem_imputed[batch_idx, tensor_indices_label]

        simulated_data_batch.append(simulated_data_label)
        original_data_batch.append(original_data_label)
        original_indices_map.append((batch_idx, tensor_indices_label))
        batch_counts.append(len(cluster_indices_in_batch))

    if not simulated_data_batch:
        continue

    # Construct entire data arrays
    start_indices = np.cumsum([0] + batch_counts[:-1])
    end_indices = np.cumsum(batch_counts)

    complete_simulated_data_for_label = np.concatenate(simulated_data_batch, axis=0)
    complete_original_data_for_label = np.concatenate(original_data_batch, axis=0)

    # Do simulation - using NumPy version, NumPy version of do_simulation ALREADY RETURNS THE DELTA FOR OPTIMIZATION PURPOSES
    result_of_sim_for_cluster_label = _do_simulation_numpy(
        coef_matrix=coef_matrix,
        simulation_input=complete_simulated_data_for_label,
        gem=complete_original_data_for_label,
        n_propagation=n_propagation
    )

    # Save the result back to the original array
    for i, (batch_idx, tensor_indices) in enumerate(original_indices_map):
        if len(tensor_indices) == 0:
            continue
        start_idx = start_indices[i]
        end_idx = end_indices[i]
        simulated_data[batch_idx, tensor_indices] = result_of_sim_for_cluster_label[start_idx:end_idx] - complete_original_data_for_label[start_idx:end_idx]