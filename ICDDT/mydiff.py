import torch.nn as nn
import torch as th
from step_sample import create_named_schedule_sampler
import numpy as np
import math
import torch
import torch.nn.functional as F
# from att_layer import * 
# from att_layer_wo_tarbeha import * 
from att_layer_wo_seqatt import *


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    从arr得到特定step的值，并广播至broadcast_shape
    """
    
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float() # 取step对应的值
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar in the limit of num_diffusion_timesteps. Beta schedules may be added, but should not be removed or changed once they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(num_diffusion_timesteps, lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,)
    elif schedule_name == 'sqrt':
        return betas_for_alpha_bar(num_diffusion_timesteps,lambda t: 1-np.sqrt(t + 0.0001),  )
    elif schedule_name == "trunc_cos":
        return betas_for_alpha_bar_left(num_diffusion_timesteps, lambda t: np.cos((t + 0.1) / 1.1 * np.pi / 2) ** 2,)
    elif schedule_name == 'trunc_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_end = scale * 0.02 + 0.01
        if beta_end > 1:
            beta_end = scale * 0.001 + 0.01
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == 'pw_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_mid = scale * 0.0001  #scale * 0.02
        beta_end = scale * 0.02
        first_part = np.linspace(beta_start, beta_mid, 10, dtype=np.float64)
        second_part = np.linspace(beta_mid, beta_end, num_diffusion_timesteps - 10 , dtype=np.float64)
        return np.concatenate([first_part, second_part])
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and produces the cumulative product of (1-beta) up to that part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):  ## 2000
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def betas_for_alpha_bar_left(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, but shifts towards left interval starting from 0
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    betas.append(min(1-alpha_bar(0), max_beta))
    for i in range(num_diffusion_timesteps-1):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)

class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)

# # TODO 3-1 历史序列编码器
# class SeqEncoder(nn.Module):
#     def __init__(self, hidden_size, args):
#         super(SeqEncoder, self).__init__()
#         self.hidden_size = hidden_size
#         self.att = Transformer_rep(args)
#         # self.dropout = nn.Dropout(args.dropout)
#         # self.norm_seq = LayerNorm(self.hidden_size)

#     def forward(self, rep_item, target_beha_rep, mask_seq):
#         output = self.att(rep_item, target_beha_rep, mask_seq)
#         # output = self.norm_seq(self.dropout(output))
#         return output # [B, L, D]

class AddLayerNorm(nn.Module):
    def __init__(self, hidden_size, args):
        super(AddLayerNorm, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(args.dropout)
        self.norm_seq = LayerNorm(self.hidden_size)
        self.linear = nn.Linear(self.hidden_size*2, 2) # 0 for scale, 1 for shift 

    def forward(self, x, target_beha_rep, t_emb):
        # t_emb = t_emb.unsqueeze(1).repeat(target_beha_rep.shape) 
        target_beha_rep = target_beha_rep[:, -1, :]
        c = self.linear(th.cat([target_beha_rep, t_emb], dim=-1))
        output = self.norm_seq(self.dropout(x))
        # print(c.shape)
        # print(output.shape)
        output = c[:, 0].unsqueeze(1).unsqueeze(2) * output + c[:, 1].unsqueeze(1).unsqueeze(2)
        return output # [B, D]

# TODO 3-1 这个是reverse阶段的输出函数
class Diffu_xstart(nn.Module):
    def __init__(self, hidden_size, args):
        super(Diffu_xstart, self).__init__()
        self.hidden_size = hidden_size
        time_embed_dim = self.hidden_size * 4
        self.time_embed = nn.Sequential(nn.Linear(self.hidden_size, time_embed_dim), SiLU(), nn.Linear(time_embed_dim, self.hidden_size))
        self.lambda_uncertainty = args.lambda_uncertainty
        self.dropout = nn.Dropout(args.dropout)
        self.norm_diffu_rep = LayerNorm(self.hidden_size)
        self.all_att = All_Transformer_rep(args)

    def modulate(self, x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def timestep_embedding(self, timesteps, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = th.exp(-math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
        if dim % 2:
            embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)

        return embedding # [B, 2D]

    def forward(self, target_beha_rep, x_t, t, item_seq_emb, mask_seq):
        '''
        x_t(融合了target和seq的表征) [B, D]
        t : [B]
        target_beha_rep [B, L, D]
        '''
        # x_t
        lambda_uncertainty = th.normal(mean=th.full(item_seq_emb.shape, self.lambda_uncertainty), std=th.full(item_seq_emb.shape, self.lambda_uncertainty)).to(x_t.device)  ## distribution
        x_t = lambda_uncertainty * x_t.unsqueeze(1) + item_seq_emb
        
        emb_t = self.time_embed(self.timestep_embedding(t, self.hidden_size)) # [B, D]
        difu_out = self.all_att(hidden=x_t, query=target_beha_rep, mask=mask_seq, condition=emb_t)
        output = self.norm_diffu_rep(self.dropout(difu_out)) 
         
        return output[:, -1, :], output 
    

class DiffuRec(nn.Module):
    def __init__(self, args,):
        super(DiffuRec, self).__init__()
        self.hidden_size = args.hidden_size
        self.schedule_sampler_name = args.schedule_sampler_name
        self.diffusion_steps = args.diffusion_steps
        
        self.gamma = args.gamma
        
        # # 这个没用到吧
        # self.use_timesteps = space_timesteps(self.diffusion_steps, [self.diffusion_steps])

        self.noise_schedule = args.noise_schedule
        betas = self.get_betas(self.noise_schedule, self.diffusion_steps)
        
        # print('betas') 16
        # print(betas.shape)
        
         # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = (betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))

        self.num_timesteps = int(self.betas.shape[0])
       
        # self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_name, self.num_timesteps)
        # TODO
        # self.aux_alpha = args.aux_alpha
        # self.tar_alpha = args.tar_alpha
        # self.aux_beta = args.aux_beta 
        # self.tar_beta = args.tar_beta
        self.aux_schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_name, self.num_timesteps, args.aux_alpha, args.aux_beta) 
        self.tar_schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_name, self.num_timesteps, args.tar_alpha, args.tar_beta) 
        
        # self.timestep_map = self.time_map() # 这个也没用到吧
        
        self.rescale_timesteps = args.rescale_timesteps
        self.original_num_steps = len(betas)

        self.xstart_model = Diffu_xstart(self.hidden_size, args)

    def get_betas(self, noise_schedule, diffusion_steps):
        ''' 取得β值'''
        betas = get_named_beta_schedule(noise_schedule, diffusion_steps)  ## array, generate beta
        return betas
    

    def q_sample(self, x_start, t, noise=None, mask=None):
        """
        前向 -- 按给定的step去扩散, 即 从 q(x_t | x_0) 采样 (reparameter trick)
        参数: x_start(x_0[B, D]); t(steps-1 [B]); noise(噪声 [B, D]); mask; 
        返回: 加噪的x_start
        """
        if noise is None:
            noise = th.randn_like(x_start) # randn_like生成与x_start相同维度的从正态分布（0,1）中采样的随机元素的张量
        
        assert noise.shape == x_start.shape # [B, 2D]
        x_t = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise  ## reparameter trick
        )  ## genetrate x_t based on x_0 (x_start) with reparameter trick
        
        # 这里默认是None的
        if mask == None:
            return x_t
        else:
            mask = th.broadcast_to(mask.unsqueeze(dim=-1), x_start.shape)  ## mask: [0,0,0,1,1,1,1,1]
            return th.where(mask==0, x_start, x_t)  ## replace the output_target_seq embedding (x_0) as x_t

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t
    
    def _predict_xstart_from_eps(self, x_t, t, eps):
        
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior: 
            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )  ## \mu_t
        assert (posterior_mean.shape[0] == x_start.shape[0])
        return posterior_mean

    def p_mean_variance(self, rep_target_beha, x_t, t, item_seq_emb, mask_seq):
        # seq_rep = rep_item[:, -1, :]
        # x_t_concat = th.cat([x_t, seq_rep], dim=-1)
        # reverse阶段应该不用再把x_t去减去历史序列表征
        model_output, _ = self.xstart_model(rep_target_beha, x_t, self._scale_timesteps(t), item_seq_emb, mask_seq)
        
        x_0 = model_output  ##output predict [B, D]
        # x_0 = self._predict_xstart_from_eps(x_t, t, model_output)  ## eps predict
        
        model_log_variance = np.log(np.append(self.posterior_variance[1], self.betas[1:]))
        model_log_variance = _extract_into_tensor(model_log_variance, t, x_t.shape)
        
        model_mean = self.q_posterior_mean_variance(x_start=x_0, x_t=x_t, t=t)  ## x_start: candidante item embedding, x_t: inputseq_embedding + outseq_noise, output x_(t-1) distribution
        return model_mean, model_log_variance

    def p_sample(self, target_beha_rep, noise_x_t, t, item_seq_emb, mask_seq):
        model_mean, model_log_variance = self.p_mean_variance(target_beha_rep, noise_x_t, t, item_seq_emb, mask_seq)
        noise = th.randn_like(noise_x_t)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(noise_x_t.shape) - 1))))  # no noise when t == 0
        sample_xt = model_mean + nonzero_mask * th.exp(0.5 * model_log_variance) * noise  ## sample x_{t-1} from the \mu(x_{t-1}) distribution based on the reparameter trick
        return sample_xt

    def reverse_p_sample(self, item_rep, target_beha_rep, noise_x_t, mask_seq, tag_beha_idx):
        device = next(self.xstart_model.parameters()).device
        
        # # TODO inference的step用期望值去reverse
        # expected_value = self.tar_alpha / (self.tar_alpha + self.tar_beta)
        # buy_step = int(self.num_timesteps * expected_value)
        # indices = list(range(buy_step))[::-1]
        
        # TODO reverse_scale steps -- gamma
        buy_step = int(self.num_timesteps * self.gamma)
        indices = list(range(buy_step))[::-1]
        
        # indices = list(range(self.num_timesteps))[::-1]
        
        for i in indices: # from T to 0, reversion iteration  
            t = th.tensor([i] * item_rep.shape[0], device=device)
            with th.no_grad():
                noise_x_t = self.p_sample(target_beha_rep, noise_x_t, t, item_rep, mask_seq)
        return noise_x_t 

    def forward(self, item_emb, target_beha_rep, item_tag, mask_seq, tag_beha_idx):    
        '''tag_beha_idx为[B]; item_tag[B, D]'''    
        # noise = th.randn_like(item_tag)
        
        # TODO 
        # print(item_tag.dtype)
        tar_idx = (tag_beha_idx == 4).nonzero(as_tuple=True)[0] # 目标行为的元素索引
        aux_idx = (tag_beha_idx != 4).nonzero(as_tuple=True)[0] # 辅助行为的元素索引
        t = th.zeros(item_emb.size(0), dtype=th.long).to(item_tag.device)
        weights = th.zeros(item_emb.size(0), dtype=th.float).to(item_tag.device)
        t_aux, weights_aux = self.aux_schedule_sampler.sample(aux_idx.shape[0], item_tag.device)
        t_tar, weights_tar = self.tar_schedule_sampler.sample(tar_idx.shape[0], item_tag.device)
        t[tar_idx], weights[tar_idx] = t_tar, weights_tar 
        t[aux_idx], weights[aux_idx] = t_aux, weights_aux
        
        x = item_tag 
        noise = th.randn_like(x)
        x_t = self.q_sample(x, t, noise=noise) 

        # pre_x_0
        pre_x_0, item_rep_out = self.xstart_model(target_beha_rep, x_t, self._scale_timesteps(t), item_emb, mask_seq)  ##output predict
        
        return pre_x_0, item_rep_out, weights, t, None # seq_rep



