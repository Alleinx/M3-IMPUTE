import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from utils.utils import get_activation

def build_adjacency_matrix(shape, edges, n_of_record):
    """
    Behavior: 
        - reconstruct the known adjacency matrix according to input argument: `edges`
        - the shape of the adjacency matrix should be (N, #Feature), where N denotes the number of records.
        - M[i][j] == 1 means the j-th value of the i-th record is known.
    """
    A = torch.zeros(shape, requires_grad=False, dtype=int)
    
    rows, cols = edges
    A[rows, cols] = 1
    
    M = torch.bitwise_or(A[:n_of_record, n_of_record:], A[n_of_record: , :n_of_record].t())
    
    return M

class Attr_Relation_Net(nn.Module):
    def __init__(self, hidden_dim=None, num_of_feature=None, device=None, drop_p=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_feature = num_of_feature
        self.device = device
        self.drop_p = drop_p
        # elements at diagonal position are 0, other positions are 1.
        self.self_mask = torch.ones(self.num_feature, self.num_feature).fill_diagonal_(0).to(self.device)
        
        # <----------------for feature relation network:----------------->
        self.phi_rm = nn.Sequential(
            nn.Linear(self.num_feature, self.hidden_dim),
            nn.GELU(),
            # nn.Dropout(p=0.2),
            nn.Linear(self.hidden_dim, self.num_feature),
            nn.GELU(),
        ).to(self.device)
        
        self.phi_rr = nn.Sequential(
            nn.Linear(self.num_feature, self.hidden_dim),
            nn.GELU(),
            # nn.Dropout(p=0.2),
        ).to(self.device)
        
        self.phi_rc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(p=self.drop_p),
        ).to(self.device)
    
    def forward(self, known_mask, obs_idx, obs_mask_idx, attr_idx_need_to_be_impute , obs_embs, feature_emb):
        # obtain obs embdedding and feature embedding according to indices.
        # shape should be: (number of pairs need to impute, dim)
        obs_h = obs_embs[obs_idx]
        target_attr_emb = feature_emb[attr_idx_need_to_be_impute]       # (n, d)

        m_i = known_mask[obs_mask_idx]
        m_j = self.self_mask[attr_idx_need_to_be_impute]

        lf_fea = target_attr_emb.unsqueeze(1)                                   # (n, 1, d)
        rt_fea_emb = (feature_emb.t()).unsqueeze(0)                             # (m, d) -> (d,m) -> (1, d, m)
        rt_fea_emb = rt_fea_emb.expand(target_attr_emb.shape[0], -1, -1)        # (n, d, m)

        # compute correlation:
        a_j_i = lf_fea @ rt_fea_emb     # (n, 1, m)
        a_j_i = a_j_i.squeeze(1)        # (n, m)

        # soft masking
        m_J_I = F.softmax(m_i * m_j, dim=1)
        m_J_I = self.phi_rm(m_J_I)                                      # (n, m)

        # soft maskout correlation
        a_j_i = self.phi_rr(a_j_i * m_J_I)                              # (n, d)

        # final mlp:
        c_r_ji = self.phi_rc(obs_h * a_j_i)

        return c_r_ji


class Similariy_Net(nn.Module):
    def __init__(self, hidden_dim=None, 
                       num_of_record=None,
                       num_of_feature=None,
                       device=None,
                       num_sample_peer=None,
                       record_data=None,
                       sample_strategy=None,
                       train_known_mask_and_attr=None,
                       cos_feature_embs=None,
                       drop_p=0.1):

        super().__init__()
        self.num_record = num_of_record
        self.hidden_dim = hidden_dim
        self.num_feature = num_of_feature
        self.device = device 
        
        # one-hot diagonal matrix, where values on diagonal are 1 and others are 0
        self.keep_mask = torch.zeros(self.num_feature, self.num_feature).fill_diagonal_(1).to(self.device)

        # for computing similarity:
        self.sim_attr_I = Attr_Relation_Net(hidden_dim, num_of_feature, device)
        self.sim_attr_PEER = Attr_Relation_Net(hidden_dim, num_of_feature, device)

        # for sample peer:
        self.sample_peer_size  = num_sample_peer
        self.sample_strategy   = sample_strategy

        if self.sample_strategy == 'cos-similarity':
            self.__construct_sample_space(record_data, train_known_mask_and_attr, cos_feature_embs)

        # <----------------for similar peer info network:----------------->
        self.drop_p = drop_p

        self.phi_sc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(p=self.drop_p),
        ).to(self.device)

        self.phi_sr = nn.Sequential(
            nn.Linear(self.num_feature * 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU()
            # nn.Dropout(p=0.1),
        ).to(self.device)

        self.phi_sm = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        ).to(self.device)

    def update_cos_prob(self, feature_node_embs):
        if hasattr(self, 'sample_prob'):
            del self.sample_prob
            torch.cuda.empty_cache()

        obs_node_embs = torch.matmul(self.reconstruct_record, feature_node_embs)             # (N, dim)
        # -----------------------------------------------------------------------------
        
        N, dim = obs_node_embs.shape                      

        lf = obs_node_embs.unsqueeze(1)       # (N, 1, dim)                       # modify shape
        lf = lf.expand(-1, N, -1)             # (N, N, dim)                       # broadcasting: expand() won't allocate new memory, it's resource efficient

        rt = obs_node_embs.unsqueeze(0)       # (1, N, dim)
        rt = rt.expand(N, -1, -1)             # (N, N, dim)

        self.sample_prob = F.cosine_similarity(lf, rt, dim=2)                                # cos shape: (N, N)
        self.sample_prob = self.sample_prob.fill_diagonal_(0)                                # mask out similarity with self

        # free GPU memory:
        del lf, rt
        torch.cuda.empty_cache()

        # ----------------------
        # if the row-wise cosine similarity is zero:
        # (Note: this case only happens when the row within the record is all 0):
        row_wise_prob_sum = torch.sum(self.sample_prob, dim=1)
        __uniform_dist_prob = torch.ones(N, N).to(self.device)
        zero_prob_position = (row_wise_prob_sum <= 0).reshape(-1, 1)
        __uniform_dist_prob = __uniform_dist_prob * zero_prob_position

        self.sample_prob += __uniform_dist_prob             

    def __construct_sample_space(self, record_data, train_mask_and_attr, feature_node_embs):
        N = self.num_feature + self.num_record
        
        all_train_known_mask, all_train_known_attr = train_mask_and_attr

        # -----------------------------------------------------------------------------
        # Same as observation node initialization:
        _build_edge_attributes = torch.zeros(N, N, requires_grad=False).to(self.device)
        rows, cols = all_train_known_mask

        _build_edge_attributes[rows, cols] = all_train_known_attr.squeeze() # fill in record value

        edge_attr_1 = _build_edge_attributes[:self.num_record, self.num_record:]
        edge_attr_2 = _build_edge_attributes[self.num_record:, :self.num_record]
        attn_score = (edge_attr_1 + edge_attr_2.t()) / 2                        # (N, feature)

        self.reconstruct_record = attn_score
        self.update_cos_prob(feature_node_embs)
    

    def sample_peer(self, KNOWN_Adjacency_matrix, N_of_Imputation, IMP_OBS_IDX):
        if self.sample_strategy is None or self.sample_strategy == 'random_sample':
            # print('[SYS]: Sample Strategy: RANDOM')
            sampled = torch.randint(0, self.num_record, (N_of_Imputation, self.sample_peer_size)).to(self.device)
        elif self.sample_strategy == 'cos-similarity':
            # print('[SYS]: Sample Strategy: COSINE SIMILARITY')
            sample_prob = self.sample_prob[IMP_OBS_IDX]
            sampled = torch.multinomial(sample_prob, self.sample_peer_size, replacement=False).to(self.device)
        else:
            raise NotImplementedError('Invalid sample strategy')

        return sampled
    
    def forward(self, M, OBS_embs, FEA_embs, IMP_OBS_index, IMP_FEA_index):
        N = IMP_OBS_index.size(0)                                   # Get the number of paris that need to be imputed.

        # get sampled peer inedx:
        PEER_index = self.sample_peer(M, N, IMP_OBS_index)             # shape: (N, sample_size)

        # for storing similarity score:
        sim_scores = torch.zeros(N, self.sample_peer_size, dtype=float, requires_grad=True).to(self.device)             # shape: (N, #peer)
        
        # 1. compute similarity score sim(x_t, x_i | j):
        for i in range(self.sample_peer_size):
            c_i = self.sim_attr_I(M, IMP_OBS_index, PEER_index[:, i], IMP_FEA_index , OBS_embs, FEA_embs).unsqueeze(1)  # shape: (N, 1, dim)
            # print(c_i.shape)
            c_t = self.sim_attr_PEER(M, PEER_index[:, i], IMP_OBS_index, IMP_FEA_index , OBS_embs, FEA_embs).unsqueeze(1)   #shape: (N, 1, dim)
            # print(c_t.shape)
            sim = c_i @ torch.permute(c_t, (0, 2, 1))       # (N, 1, dim) x (N, dim, 1) => (N, 1, 1)
            # format sim and store it into sim_scores:
            sim = sim.squeeze(1)                            # (N, 1)
            sim_scores[:, i] = sim.squeeze(1)               # (N)
        
        _sim_v = sim_scores                         # (N, sample_size), for computing confidence
        sim_scores = sim_scores.unsqueeze(1)        # (N, 1, sample_size)

        h = OBS_embs[PEER_index]                    # (N, sample_size, dim)

        # compute r_{t | j}:
        m_p_t = M[PEER_index]                                              # (N, sample_size, A)
        m_k_j = self.keep_mask[IMP_FEA_index]                              # (N, A)
        m_k_j = m_k_j.unsqueeze(1).expand(-1, self.sample_peer_size, -1)   # (N, sample_size, A)
        m_q = torch.cat((m_p_t, m_k_j), dim=2)                             # (N, sample_size, 2A)
        # get r_{t | j}:
        r_t_j = self.phi_sr(m_q)                                            # (N, sample_size, dim)

        # feed into nn
        h_query = self.phi_sm(h * r_t_j)                                    # (N, sample_size, dim)

        # compute e^{s}_{ij}
        e_sim = sim_scores.float() @ h_query                                # (N, 1, sample_size) x (N, sample_size, dim) -> (N, 1, dim)
        e_sim = e_sim.squeeze(1)                                            # (N, dim)
        # compute c^{s}_{j <- i}
        c_sim = self.phi_sc(e_sim)                                          # (N, dim)

        return c_sim, _sim_v


class ImputeNet(nn.Module):
    def __init__(self, hidden_dim=None, 
                       num_of_record=None,
                       num_of_feature=None,
                       device=None,
                       num_sample_peer=None,
                       apply_peer=False,
                       apply_relation=False,
                       record_data=None,
                       sample_strategy=None,
                       train_known_mask_and_attr=None,
                       cos_sample_feature_embs=None,
                       drop_p=0.1):

        super().__init__()
        self.num_record = num_of_record
        self.hidden_dim = hidden_dim
        self.num_feature = num_of_feature
        self.device = device
        self.num_sample_peer = num_sample_peer

        # applied unit :
        self.apply_peer     = apply_peer
        self.apply_relation = apply_relation 
        
        self.attr_relation_net = Attr_Relation_Net(hidden_dim, num_of_feature, device, drop_p=drop_p)
        self.sim_info_net = Similariy_Net(hidden_dim=hidden_dim, 
                                          num_of_record=num_of_record,
                                          num_of_feature=num_of_feature,
                                          device=device,
                                          num_sample_peer=num_sample_peer,
                                          record_data=record_data,
                                          sample_strategy=sample_strategy,
                                          train_known_mask_and_attr=train_known_mask_and_attr,
                                          cos_feature_embs=cos_sample_feature_embs,
                                          drop_p=drop_p)
        
        # <----------------for Imputation network:----------------->
        self.sim_conf_net = nn.Sequential(
            nn.Linear(self.num_sample_peer, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        ).to(self.device)

        self.post_neural_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            # nn.Dropout(p=0.2),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        ).to(self.device)
    
    def get_imp_obs_fea_pair_index(self, impute_target):
        # 找到obs_node index 和 需要填的attribute的index
        end1, end2 = impute_target
        idx  = torch.ge(end1, self.num_record)
        idx = torch.nonzero(idx).squeeze(dim=1)[0].item()

        obs_idx                    = torch.cat((end1[:idx], end2[idx:]), dim=0).to(self.device)
        attr_idx_need_to_be_impute = torch.cat((end2[:idx], end1[idx:]), dim=0) - self.num_record
        attr_idx_need_to_be_impute.to(self.device)

        return obs_idx, attr_idx_need_to_be_impute

    def sim_confidence(self, similarity):
        # similiary shape: (N, #peer)
        similarity = torch.abs(similarity.float())
        z = self.sim_conf_net(similarity)

        z = 1 - (1 / ( torch.exp(torch.abs(z)) + 1e-4 ) )      # a new activation function.

        return z
    
    def forward(self, obs_nodes_embs, fea_nodes_embs, known_edges, impute_target_edges):
        # shape for building known_mask adjacency 
        size = (self.num_record + self.num_feature, self.num_record + self.num_feature)

        # Build known mask adjacency        
        # mask value: 1 if x_ij is known, else 0;
        known_M = build_adjacency_matrix(size, known_edges, self.num_record).to(self.device)    # known_M shape: (N, Feature)

        # obtain the index for observation nodes and corresponding imputing attributes
        IMP_OBS_IDX, IMP_FEA_IDX = self.get_imp_obs_fea_pair_index(impute_target_edges)         # (N, 1) for each IDX; N denotes the number of attributes that need to be imputed;

        if self.apply_relation:
            c_r = self.attr_relation_net(known_M, IMP_OBS_IDX, IMP_OBS_IDX, IMP_FEA_IDX, obs_nodes_embs, fea_nodes_embs)    # c_r: (N, dim)

        if self.apply_peer:
            c_s, similarity = self.sim_info_net(known_M, obs_nodes_embs, fea_nodes_embs, IMP_OBS_IDX, IMP_FEA_IDX)      # c_s: (N, dim); similairty: (N, sample_size)
        
        if not self.apply_relation:
            # print('apply peer')
            imputed_value = self.post_neural_net(c_s)
        elif not self.apply_peer:
            # print('apply relation')
            imputed_value = self.post_neural_net(c_r)
        else:
            # print('apply both')
            sim_conf = self.sim_confidence(similarity)
            imputed_value = self.post_neural_net((1 - sim_conf) * c_r + sim_conf * c_s)
        
        
        return imputed_value, 0