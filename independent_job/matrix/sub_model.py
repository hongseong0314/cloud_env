import torch
import torch.nn as nn
import torch.nn.functional as F

class AddAndInstanceNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        # input.shape: (B, T, embedding)

        added = input1 + input2
        # shape:  (B, T, embedding)

        if added.size(1) == 1:
            back_trans = added
        else:
            transposed = added.transpose(1, 2)
            # shape:  (B, embedding, T)
            normalized = self.norm(transposed)

            back_trans = normalized.transpose(1, 2)
            # shape:  (B, T, embedding)

        return back_trans


class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (B, T, embedding)

        return self.W2(F.relu(self.W1(input1)))

class Depth_MultiHeadAttention(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        head_num = model_params['head_num']
        depth_hidden_dim = model_params['depth_hidden_dim']
        depth_init = model_params['depth__init']
        FC_init = model_params['FC_init']
        self.problem_size  = 3
        Wqkv = torch.torch.distributions.Uniform(low=-depth_init, high=depth_init).sample((head_num, self.problem_size, depth_hidden_dim))
        Bqkv = torch.torch.distributions.Uniform(low=-depth_init, high=depth_init).sample((head_num, depth_hidden_dim))
        self.Wqkv = nn.Parameter(Wqkv)
        self.Bqkv = nn.Parameter(Bqkv)
        self.Wq = nn.Linear(depth_hidden_dim,depth_hidden_dim, bias=False)
        self.Wk = nn.Linear(depth_hidden_dim,depth_hidden_dim, bias=False)
        self.Wv = nn.Linear(depth_hidden_dim,depth_hidden_dim, bias=False)
        # shape: (head, 2, depth_hidden)

        FC_weight = torch.torch.distributions.Uniform(low=-FC_init, high=FC_init).sample((head_num, depth_hidden_dim, 1))
        FC_bias = torch.torch.distributions.Uniform(low=-FC_init, high=FC_init).sample((head_num, 1))
        self.FC_weight = nn.Parameter(FC_weight)
        # shape: (head, ms_hidden, 1)
        self.FC_bias = nn.Parameter(FC_bias)
        # shape: (head, 1)

    def forward(self, q, k, v, D_TM):
        # q shape: (B, H, T, qkv_dim)
        # k,v shape: (B, H, M, qkv_dim)
        # D_TM.shape: (B, T, M)
        batch_size = q.size(0)
        T_cnt = q.size(2)
        M_cnt = k.size(2)

        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        sqrt_qkv_dim = self.model_params['sqrt_qkv_dim']

        dot_product = torch.matmul(q, k.transpose(2, 3))
        dot_product_score = dot_product / sqrt_qkv_dim

        D_TM = D_TM[:, None, :, :, :].expand(batch_size, head_num, T_cnt, M_cnt, self.problem_size-1 )
        # [B, H, T, M]

        a_e_feature  = torch.cat((dot_product_score[..., None], D_TM), dim=4)
        # [B, H, T, M, 2]
        a_e_feature_transposed = a_e_feature.transpose(1,2)
        # [B, T, H, M, 2]
        depth_qkv = torch.matmul(a_e_feature_transposed, self.Wqkv)
        depth_qkv = F.leaky_relu(depth_qkv + self.Bqkv[None, None, :, None, :])

        depth_q = self.Wq(depth_qkv)
        depth_k = self.Wk(depth_qkv)
        depth_v = self.Wv(depth_qkv)
        # [B, T, H, M, depth_dim]

        depth_dot_product = torch.matmul(depth_q, depth_k.transpose(3, 4))
        # depth_dot_product = depth_dot_product / sqrt_qkv_dim
        # [B ,T, H, M, M]

        depth_weights = nn.Softmax(dim=4)(depth_dot_product)
        ae_score = torch.matmul(depth_weights, depth_v)
        ae_score = ae_score + depth_qkv
        # [B, T, H, M, depth_dim]

        fc_out = torch.matmul(ae_score, self.FC_weight)
        fc_out = fc_out + self.FC_bias[None, None, :, None, :]
        depth_scores = fc_out.transpose(1,2)
        # [B, H, T, M, 1]
        depth_scores = depth_scores.squeeze(-1)
        # [B, H, T, M]

        weights = nn.Softmax(dim=3)(depth_scores)
        # [B, H, T, M]

        out = torch.matmul(weights, v)
        # [B, H, T, qkv_dim]
        out_transposed = out.transpose(1, 2)
        # [B, T, H, qkv_dim]
        out_concat = out_transposed.reshape(batch_size, T_cnt, head_num * qkv_dim)
        # [B, T, head*qkv_dim]

        return out_concat

class MixedScore_MultiHeadAttention(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        head_num = model_params['head_num']
        ms_hidden_dim = model_params['ms_hidden_dim']
        mix1_init = model_params['ms_layer1_init']
        mix2_init = model_params['ms_layer2_init']
        self.problem_size = 3
        mix1_weight = torch.torch.distributions.Uniform(low=-mix1_init, high=mix1_init).sample((head_num, self.problem_size , ms_hidden_dim))
        mix1_bias = torch.torch.distributions.Uniform(low=-mix1_init, high=mix1_init).sample((head_num, ms_hidden_dim))
        self.mix1_weight = nn.Parameter(mix1_weight)
        # shape: (head, 2, ms_hidden)
        self.mix1_bias = nn.Parameter(mix1_bias)
        # shape: (head, ms_hidden)

        mix2_weight = torch.torch.distributions.Uniform(low=-mix2_init, high=mix2_init).sample((head_num, ms_hidden_dim, 1))
        mix2_bias = torch.torch.distributions.Uniform(low=-mix2_init, high=mix2_init).sample((head_num, 1))
        self.mix2_weight = nn.Parameter(mix2_weight)
        # shape: (head, ms_hidden, 1)
        self.mix2_bias = nn.Parameter(mix2_bias)
        # shape: (head, 1)

    def forward(self, q, k, v, D_TM):
        # q shape: (B, H, T, qkv_dim)
        # k,v shape: (B, H, M, qkv_dim)
        # D_TM.shape: (B, T, M)
        batch_size = q.size(0)
        T_cnt = q.size(2)
        M_cnt = k.size(2)

        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        sqrt_qkv_dim = self.model_params['sqrt_qkv_dim']

        dot_product = torch.matmul(q, k.transpose(2, 3))
        dot_product_score = dot_product / sqrt_qkv_dim
        # print("dot product shape :",dot_product_score.shape)
        D_TM = D_TM[:, None, :, :, :].expand(batch_size, head_num, T_cnt, M_cnt, self.problem_size-1 )
        # D_TM = D_TM[:, None, :, :].expand(batch_size, head_num, T_cnt, M_cnt)
        # [B, H, T, M]
        # print("D_ij shape :",D_TM.shape)

        two_scores  = torch.cat((dot_product_score[..., None], D_TM), dim=4)
        # [B, H, T, M, 2]
        two_scores_transposed = two_scores.transpose(1,2)
        # [B, T, H, M, 2]
        # print("two_scores shape :",two_scores_transposed.shape)

        ms1 = torch.matmul(two_scores_transposed, self.mix1_weight)
        ms1 = ms1 + self.mix1_bias[None, None, :, None, :]
        ms1_activated = F.relu(ms1)
        # [B, T, H, M, mix_dim]

        ms2 = torch.matmul(ms1_activated, self.mix2_weight)
        ms2 = ms2 + self.mix2_bias[None, None, :, None, :]
        mixed_scores = ms2.transpose(1,2)
        # [B, H, T, M, 1]
        mixed_scores = mixed_scores.squeeze(-1)
        # [B, H, T, M]
        # print("mixed_scores shape :",mixed_scores.shape)

        weights = nn.Softmax(dim=3)(mixed_scores)
        # [B, H, T, M]
        # print("weights shape :",weights.shape)

        out = torch.matmul(weights, v)
        # [B, H, T, qkv_dim]
        out_transposed = out.transpose(1, 2)
        # [B, T, H, qkv_dim]
        out_concat = out_transposed.reshape(batch_size, T_cnt, head_num * qkv_dim)
        # [B, T, head*qkv_dim]

        return out_concat

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, device):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.device = device

    def pre_set(self, task_num):
        self.encoding = torch.zeros(1, task_num, self.d_model, device=self.device)
        self.encoding.requires_grad = False

        pos = torch.arange(0, task_num, device =self.device)
        pos = pos.float().unsqueeze(dim=1)
        _2i = torch.arange(0, self.d_model, step=2, device=self.device).float()

        self.encoding[:, :, ::2] = torch.sin(pos / (10000 ** (_2i / self.d_model)))
        self.encoding[:, :, 1::2] = torch.cos(pos / (10000 ** (_2i / self.d_model)))

    def forward(self, available_task):
        return self.encoding[:, available_task, :]