import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from utils.utils import get_activation

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
    
    def forward(self, known_mask, obs_idx, obs_mask_idx, attr_idx_need_to_be_impute, obs_embs, fea_corr):
        # obtain obs embdedding and feature embedding according to indices.
        # shape should be: (number of pairs need to impute, dim)
        obs_h = obs_embs[obs_idx]

        m_i = known_mask[obs_mask_idx]
        m_j = self.self_mask[attr_idx_need_to_be_impute]

        # get feature correlation:
        a_j_i = fea_corr[attr_idx_need_to_be_impute]        # (n, m)

        # soft masking
        m_J_I = F.softmax(m_i * m_j, dim=1)
        m_J_I = self.phi_rm(m_J_I)                                      # (n, m)

        # soft maskout correlation
        a_j_i = self.phi_rr(a_j_i * m_J_I)                              # (n, d)

        # final mlp:
        c_r_ji = self.phi_rc(obs_h * a_j_i)

        return c_r_ji

class Attr_Relation_Net_SRU(nn.Module):
    """
    A variant of class Attr_Relation_Net, optimized for the SRU unit
    """
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
    
    def forward(self, obs_h, known_mask,  attr_idx_need_to_be_impute, fea_corr):
        #obs_h: (n, k, d)
        n, k, m = known_mask.shape

        m_i = known_mask                                                # (n, k, d)
        m_j = self.self_mask[attr_idx_need_to_be_impute]                # (n, m)
        m_j = m_j.unsqueeze(1).expand(-1, k, -1)                        # (n, k, m)

        # # compute correlation:
        a_j_i = fea_corr[attr_idx_need_to_be_impute]    # (n, m)
        a_j_i = a_j_i.unsqueeze(1)                      # (n, 1, m)
        a_j_i = a_j_i.expand(-1, k, -1)                 # (n, K, m)

        # soft masking
        m_J_I = F.softmax(m_i * m_j, dim=2)                             # (n, k, m) * (n, k, m)
        m_J_I = self.phi_rm(m_J_I)                                      # (n, k, m)

        # soft maskout correlation
        a_j_i = self.phi_rr(a_j_i * m_J_I)                              # (n, k, d)

        # final mlp:
        c_r_ji = self.phi_rc(obs_h * a_j_i)                             # (n, k, d) * (n, k, d) => (n, k, d)

        return c_r_ji

class Similariy_Net(nn.Module):
    def __init__(self, hidden_dim=None, 
                       num_of_record=None,
                       num_of_feature=None,
                       device=None,
                       num_sample_peer=None,
                       record_data=None,
                       sample_strategy=None,
                       cos_feature_embs=None,
                       drop_p=0.1,
                       running_on_very_large_dataset=False,
                       sample_space_size=None,
                       attn_score=None):

        super().__init__()
        self.num_record = num_of_record
        self.hidden_dim = hidden_dim
        self.num_feature = num_of_feature
        self.device = device 
        
        # one-hot diagonal matrix, where values on diagonal are 1 and others are 0
        self.keep_mask = torch.zeros(self.num_feature, self.num_feature).fill_diagonal_(1).to(self.device)

        # for computing similarity:
        self.sim_attr_I = Attr_Relation_Net_SRU(hidden_dim, num_of_feature, device)
        self.sim_attr_PEER = Attr_Relation_Net_SRU(hidden_dim, num_of_feature, device)

        # for sample peer:
        self.sample_peer_size  = num_sample_peer
        self.sample_strategy   = sample_strategy

        # For very Large dataset:
        self.running_on_large_dataset = running_on_very_large_dataset
        if self.running_on_large_dataset:
            print(f'[Config] Running on very large dataset = {self.running_on_large_dataset}')
            self.sample_space_size = sample_space_size

        if self.sample_strategy == 'cos-similarity':
            self.__construct_sample_space(attn_score, cos_feature_embs)

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

    def very_large_dataset_update_sample_space(self):
        """
        create the sample space with size (K,)
        this is only applied when running on very large dataset
        """
        if hasattr(self, 'sample_space'):
            del self.sample_space

        self.sample_space = torch.randint(0, self.num_record, (self.sample_space_size, )).to(self.device)      # (K,), K = sample space size

    def update_cos_prob(self, feature_node_embs, attn_score):
        if hasattr(self, 'sample_prob'):
            del self.sample_prob
            torch.cuda.empty_cache()

        if self.running_on_large_dataset:
            self.very_large_dataset_update_sample_space()
            lf = torch.matmul(attn_score, feature_node_embs)                                           # (N, dim)

            sample_emb = torch.matmul(attn_score[self.sample_space], feature_node_embs)             # (K, dim)
            # -----------------------------------------------------------------------------
            N, dim = lf.shape                  

            lf = lf.unsqueeze(1)                            # (N, 1, dim)                       
            lf = lf.expand(-1, self.sample_space_size, -1)  # (N, K, dim)                           
            rt = sample_emb.unsqueeze(0)                    # (1, K, dim)
            rt = rt.expand(N, -1, -1)                       # (N, K, dim)

            self.sample_prob = F.cosine_similarity(lf, rt, dim=2)                                # cos shape: (N, K)
        else:
            obs_node_embs = torch.matmul(attn_score, feature_node_embs)             # (N, dim)
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

    def __construct_sample_space(self, attn_score, feature_node_embs):
        # self.reconstruct_record = attn_score
        self.update_cos_prob(feature_node_embs, attn_score)
    

    def sample_peer(self, KNOWN_Adjacency_matrix, N_of_Imputation, IMP_OBS_IDX):
        if self.sample_strategy is None or self.sample_strategy == 'random_sample':
            # print('[SYS]: Sample Strategy: RANDOM')
            sampled = torch.randint(0, self.num_record, (N_of_Imputation, self.sample_peer_size)).to(self.device)
        elif self.sample_strategy == 'cos-similarity':
            # print('[SYS]: Sample Strategy: COSINE SIMILARITY')
            sample_prob = self.sample_prob[IMP_OBS_IDX]
            sampled = torch.multinomial(sample_prob, self.sample_peer_size, replacement=False).to(self.device)

            if self.running_on_large_dataset:
                # convert back to original node index
                sampled = self.sample_space[sampled]
        else:
            raise NotImplementedError('Invalid sample strategy')

        return sampled
    
    def forward(self, M, OBS_embs, IMP_OBS_index, IMP_FEA_index, fea_corr):
        # M: (n, m)
        N = IMP_OBS_index.size(0)                                   # Get the number of paris that need to be imputed.
        K = self.sample_peer_size
        # get sampled peer inedx:
        PEER_index = self.sample_peer(M, N, IMP_OBS_index)             # shape: (N, K)
        
        peer_mask = M[PEER_index]   # (N, K, m)
        peer_hidden = OBS_embs[PEER_index] # (N, K, d)

        sample_hidden = OBS_embs[IMP_OBS_index]                     # (N, d)
        sample_hidden = sample_hidden.unsqueeze(1)                  # (impute_num, 1, d)
        sample_hidden = sample_hidden.expand(-1, K, -1)             # (impute_num, K, d)

        sample_mask = M[IMP_OBS_index]                              # (impute_num, m)
        sample_mask = sample_mask.unsqueeze(1).expand(-1, K, -1)    # (impute_num, K, m)

        c_i = self.sim_attr_I(sample_hidden, peer_mask, IMP_FEA_index, fea_corr)    # (N, K, d)
        c_t = self.sim_attr_PEER(peer_hidden, sample_mask, IMP_FEA_index, fea_corr) # (N, K, d)
        sim_scores = torch.cosine_similarity(c_i, c_t, dim=2)                       # (N, K)
        
        # block the gradient backward here:
        # _sim_v = sim_scores.clone().detach().to(self.device)                       # (N, sample_size), for computing confidence
        # keep the gradient backward here:
        _sim_v = sim_scores                                                          # (N, sample_size), for computing confidence
        sim_scores = sim_scores.unsqueeze(1)                                         # (N, 1, sample_size)

        # compute r_{t | j}:
        h = peer_hidden                                                     # (N, sample_size, dim)
        m_p_t = peer_mask                                                   # (N, sample_size, A)
        m_k_j = self.keep_mask[IMP_FEA_index]                               # (N, A)
        m_k_j = m_k_j.unsqueeze(1).expand(-1, self.sample_peer_size, -1)    # (N, sample_size, A)
        m_q = torch.cat((m_p_t, m_k_j), dim=2)                              # (N, sample_size, 2A)
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
                       cos_sample_feature_embs=None,
                       drop_p=0.1,
                       running_on_very_large_dataset=False,
                       sample_space_size=None,
                       attn_score=None):

        super().__init__()
        self.num_record = num_of_record
        self.hidden_dim = hidden_dim
        self.num_feature = num_of_feature
        self.device = device
        self.num_sample_peer = num_sample_peer

        # applied unit :
        self.apply_peer     = apply_peer
        self.apply_relation = apply_relation 
        
        self.FCU = Attr_Relation_Net(hidden_dim, num_of_feature, device, drop_p=drop_p)
        self.SRU = Similariy_Net(hidden_dim=hidden_dim, 
                                          num_of_record=num_of_record,
                                          num_of_feature=num_of_feature,
                                          device=device,
                                          num_sample_peer=num_sample_peer,
                                          record_data=record_data,
                                          sample_strategy=sample_strategy,
                                          cos_feature_embs=cos_sample_feature_embs,
                                          drop_p=drop_p,
                                          running_on_very_large_dataset=running_on_very_large_dataset,
                                          sample_space_size=sample_space_size,
                                          attn_score=attn_score)
        
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
        # obtain obs_node index and target imputation feature index
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
    
    def forward(self, obs_nodes_embs, fea_nodes_embs, known_edges, impute_target_edges, known_mask=None):
        # obtain the index for observation nodes and corresponding imputing attributes
        IMP_OBS_IDX, IMP_FEA_IDX = self.get_imp_obs_fea_pair_index(impute_target_edges)         # (N, 1) for each IDX; N denotes the number of attributes that need to be imputed;
        # compute feature correlations once and reuse for this epoch/forward run:
        feature_corr = fea_nodes_embs @ fea_nodes_embs.t()                                      # (m, m)

        if self.apply_relation:
            c_r = self.FCU(known_mask, IMP_OBS_IDX, IMP_OBS_IDX, IMP_FEA_IDX, obs_nodes_embs, feature_corr)    # c_r: (N, dim)

        if self.apply_peer:
            c_s, similarity = self.SRU(known_mask, obs_nodes_embs, IMP_OBS_IDX, IMP_FEA_IDX, feature_corr)      # c_s: (N, dim); similairty: (N, sample_size)

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