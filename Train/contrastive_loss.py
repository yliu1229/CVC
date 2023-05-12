import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from Utils.utils import calc_topk_accuracy


class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature  # 0.5
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss()

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), device=self.device)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N, device=self.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)

        top1, top3 = calc_topk_accuracy(logits, labels, (1, 3))

        del sim, sim_i_j, sim_j_i

        return loss, top1, top3


class EntropyLoss(nn.Module):
    def __init__(self, batch_size, cluster_num):
        super(EntropyLoss, self).__init__()
        self.batch_size = batch_size
        self.cluster_num = cluster_num

    def forward(self, cluster_matrix):
        """
        This regularization loss function makes the following assumption:
        cluster_num (i.e. the num of columns of matrix) >> batch_size (i.e. the num of rows)
        1. Each row in cluster_matrix should be one-hot, which makes the entropy of cluster
        assignment probabilities of each sample minimum (ideally 0);
        2. each sample should be assign to different cluster.
        :param cluster_matrix: shape = [B, cluster_num]
        """
        h_cluster = - cluster_matrix * torch.log(cluster_matrix + 1e-8)
        h_cluster = h_cluster.sum() / h_cluster.size(0)  # averaged entropy of clusters

        cluster_sum = cluster_matrix.sum(dim=0)
        cluster_sum /= self.batch_size
        scatter = cluster_sum * torch.log(cluster_sum + 1e-8)
        scatter = scatter.sum() + math.log(self.cluster_num)

        loss = h_cluster + scatter
        return loss, scatter


class ClusterLoss(nn.Module):
    def __init__(self, memory_bank, temperature):
        super(ClusterLoss, self).__init__()
        self.memory = memory_bank
        self.temperature = temperature

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, instance_m, cluster_m, penalize_instance_indices, penalize_cluster_indices):

        cluster_max, indices = cluster_m.max(dim=1)

        cluster_indices = []
        instance_indices = []
        for i in range(cluster_m.size(0)):
            if cluster_max[i] >= 0.8:
                cluster_indices.append(indices[i])
                instance_indices.append(i)

        # Re-clustering penalized instances
        if len(penalize_instance_indices) != 0:
            instance_indices += penalize_instance_indices
            cluster_indices += penalize_cluster_indices

        loss = top1 = top3 = 0
        if len(cluster_indices) != 0:
            print('\tComparing %d entry' % len(cluster_indices), end=';')
            cluster_indices = torch.stack(cluster_indices)

            sim = torch.matmul(instance_m[instance_indices], self.memory.T) / self.temperature
            loss = self.criterion(sim, cluster_indices)

            top1, top3 = calc_topk_accuracy(sim, cluster_indices, (1, 3))
        else:
            print('\tWarning: no cluster comparing happened!')

        del sim, cluster_max, indices, cluster_indices, instance_indices

        return loss, top1, top3


class MemoryManager:
    def __init__(self, memory_bank, update_ratio=0.1):
        self.memory = memory_bank
        self.update_ratio = update_ratio

    def setup_memory(self, instance_m, cluster_m):
        # Only store cluster feature when probability >= 0.9
        features = instance_m.detach()
        cluster_max, indices = cluster_m.max(dim=1)

        cluster_indices = []
        instance_indices = []
        for i in range(cluster_m.size(0)):
            if cluster_max[i] >= 0.95:
                cluster_indices.append(indices[i])
                instance_indices.append(i)

        if len(cluster_indices) != 0:
            print('\tADDing %d entry' % len(cluster_indices), end=';')
            cluster_indices = torch.stack(cluster_indices)
            self.memory[cluster_indices] = features[instance_indices]

        del features, cluster_m, indices, cluster_indices, instance_indices

    def update_memory_bank(self, instance_m, cluster_m):
        """
            update cluster only when confidence >=0.9
            if more than one instance (>0.9) are ready for update, only the latter one is used
        """
        # Detach from computational graph
        instance_m = instance_m.detach()
        cluster_max, indices = cluster_m.max(dim=1)

        cluster_indices = []
        instance_indices = []
        for i in range(cluster_m.size(0)):
            if cluster_max[i] >= 0.95:
                cluster_indices.append(indices[i])
                instance_indices.append(i)

        # For any cluster==0, store index:instance feature
        # SHOULD NOT BE USED
        zero_cluster_indices = []
        zero_instance_indices = []
        for index, i in enumerate(cluster_indices):
            if torch.equal(self.memory[i], torch.zeros(self.memory[i].shape, device=self.memory.device)):
                zero_cluster_indices.append(i)
                zero_instance_indices.append(instance_indices[index])
        if len(zero_cluster_indices) != 0:
            print('\tMemory update: zero re-set num = ', len(zero_cluster_indices))
            zero_cluster_indices = torch.stack(zero_cluster_indices)

        # Update memory bank with ratio
        if len(cluster_indices) != 0:
            # print('\tUPDATing %d entry to memory bank...' % len(cluster_indices))
            cluster_indices = torch.stack(cluster_indices)
            self.memory[cluster_indices] -= self.update_ratio * (self.memory[cluster_indices] - instance_m[instance_indices])

        # Use stored instance feature to set up cluster==0
        if len(zero_cluster_indices) != 0:
            self.memory[zero_cluster_indices] = instance_m[zero_instance_indices]

        del instance_m, cluster_m, indices, cluster_indices, instance_indices, \
            zero_instance_indices, zero_cluster_indices,
