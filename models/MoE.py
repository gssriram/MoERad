import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.nn import functional as F
from models.BCL import BCLModel
from torchvision import transforms
from PIL import Image

class MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k

        # Experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.n_embd, 4*config.n_embd),
                nn.GELU(approximate='tanh'),
                nn.Linear(4*config.n_embd, config.n_embd)
            ) for _ in range(config.num_experts)
        ])

        # Gating
        self.gate = nn.Linear(config.n_embd, config.num_experts)
        self.gate.weight.data.normal_(std=0.02/config.num_experts)

        self.register_buffer('expert_biases', torch.zeros(config.num_experts))
        self.register_buffer('expert_counts', torch.zeros(config.num_experts))
        self.bias_update_rate = config.bias_update_rate
        self._current_batch_count = None

    def forward(self, x, attn_mask=None):
        # print(x.shape)
        B, T, D = x.size()
        x_flat = x.view(-1, D)  # [B*T, D]

        if attn_mask is None:
            gate_scores = torch.sigmoid(self.gate(x_flat))
            _, topk_indices = torch.topk(gate_scores, self.top_k, dim=-1)  # [B*T, K]
        else:
            mask_flat = attn_mask.view(-1).bool()

            # Initialize full-size gate_scores tensor
            # print(x.dtype)
            gate_scores = torch.zeros(B*T, self.num_experts, 
                                    dtype=torch.bfloat16, 
                                    device=x.device)

            # Process only non-pad tokens
            non_pad_indices = torch.where(mask_flat)[0]  # More reliable than torch.nonzero()

            if non_pad_indices.numel() > 0:
                x_non_pad = x_flat[non_pad_indices]  # [N, D]
                
                # Compute scores for non-pad tokens
                gate_scores_non_pad = torch.sigmoid(self.gate(x_non_pad)).to(dtype=torch.bfloat16)
                # print(gate_scores.dtype, non_pad_indices.dtype, gate_scores_non_pad.dtype)
                gate_scores[non_pad_indices] = gate_scores_non_pad

            gate_logits = gate_scores + self.expert_biases
            _, topk_indices = torch.topk(gate_logits, self.top_k, dim=-1)  # [B*T, K]
        
        # We are using the weights that are unaffected by bias correction
        topk_weights = gate_scores.gather(-1, topk_indices)

        self._current_batch_count = torch.bincount(topk_indices.flatten(), minlength=self.num_experts)
        
        # Sparse routing
        expert_outputs = torch.zeros_like(x_flat, dtype=gate_scores.dtype)
        for expert_idx in range(self.num_experts):
            mask = topk_indices == expert_idx
            positions = torch.nonzero(mask)
            
            if positions.shape[0] == 0:
                continue
                
            # Get weights and inputs
            token_indices = positions[:,0]
            expert_weights = topk_weights[mask].unsqueeze(-1)  # [N,1]
            expert_input = x_flat[token_indices]  # [N,D]
            
            # Process and accumulate
            expert_out = self.experts[expert_idx](expert_input)
            expert_outputs.index_add_(0, token_indices, expert_weights * expert_out)

        return expert_outputs.view(B,T,D), gate_scores.view(B,T,self.num_experts)

    def bias_update(self):
        if self._current_batch_count is None:
            return
        total_tokens = self._current_batch_count.sum().item()
        avg_load = total_tokens / self.num_experts

        # e_i = avg(c_i) - c_i
        load_violation = avg_load - self._current_batch_count
        delta = self.bias_update_rate * torch.sign(load_violation)

        #  b_i = b_i + u ∗ sign(e_i)
        self.expert_biases += delta.to(self.expert_biases.device)
        self.expert_counts += self._current_batch_count

        # print(self.expert_counts)
        # reseting
        self._current_batch_count = None

    def print_details(self):
        print(self.expert_counts.cpu().numpy())

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class VisualSelfAttention(nn.Module):
    def __init__(self, config, dropout_p):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout_p = dropout_p

    def forward(self, x): # (B, T, n_embd)   
        # print(x.shape)     
        B, T, C = x.size()

        qkv = self.c_attn(x) #(B, 256, 2304)
        q, k, v = qkv.split(self.n_embd, dim=2) # (B, T, n_embd), (B, T, n_embd), (B, T, n_embd)
        # print(q.shape, k.shape, v.shape)
        
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # print(q.shape, k.shape, v.shape)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=(self.dropout_p if self.training else 0.0)) # flash attention (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # (B, T, n_embd)
        y = self.c_proj(y) # (B, T, n_embd)
        # print(y.shape)
        # raise KeyboardInterrupt
        return y

class CrossAttention(nn.Module):
    def __init__(self, config, dropout_p):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn_kv = nn.Linear(config.n_embd, 2 * config.n_embd)
        self.c_attn_q = nn.Linear(config.n_embd, config.n_embd)
        
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout_p = dropout_p
        

    def forward(self, x, query):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        B_q, T_q, C_q = query.size()
        # kv = self.c_attn(x)
        # k, v = kv.split(self.n_embd, dim=2)
        
        kv = self.c_attn_kv(x)
        k, v = kv.split(self.n_embd, dim=2)
        q = self.c_attn_q(query)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B_q, T_q, self.n_head, C_q // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # print(q.shape, k.shape, v.shape)

        y = F.scaled_dot_product_attention(q, k, v, dropout_p=(self.dropout_p if self.training else 0.0)) # flash attention
        y = y.transpose(1, 2).contiguous().view(B_q, T_q, C_q) # re-assemble all head outputs side by side
        # print(f'At cross-attn before MLP: {y.shape}')
        # output projection
        y = self.c_proj(y)
        # print(f'At cross-attn after MLP: {y.shape}')
        return y

class CasualSelfAttention(nn.Module):
    def __init__(self, config, dropout_p):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout_p = dropout_p

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # print(x.shape)
        qkv = self.c_attn(x)
        # print(qkv.shape)
        q, k, v = qkv.split(self.n_embd, dim=2) # (B, T, n_embd), (B, T, n_embd), (B, T, n_embd)
        # print(q.shape, k.shape, v.shape)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # print(q.shape, k.shape, v.shape)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=(self.dropout_p if self.training else 0.0)) # flash attention
        # print(y.shape)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # print(y.shape)
        # output projection
        y = self.c_proj(y)
        # print(f'Casual self attn.: {y.shape}')
        # raise KeyboardInterrupt
        return y

class Block(nn.Module):
    def __init__(self, config, dropout_p):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config, dropout_p)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.cross_attn = CrossAttention(config, dropout_p)
        self.ln3 = nn.LayerNorm(config.n_embd)
        self.moe_layer = MoE(config)

    def forward(self, x, imgs, dis_logits, attn_mask=None): # (B, T, n_embd)
        x = x + self.attn(self.ln1(x)) # (B, T, n_embd)
        x = x + self.cross_attn(self.ln2(imgs), self.ln2(x)) # (B, T, n_embd)
        x_moe, _ = self.moe_layer(self.ln3(x), attn_mask)
        x = x + x_moe
        return x

class EncoderBlock(nn.Module): 
    def __init__(self, config, dropout_p):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = VisualSelfAttention(config, dropout_p)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 256 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 6 # number of layers
    n_head: int = 8 # number of heads
    n_embd: int = 768 # embedding dimension
    num_experts: int = 32
    top_k: int = 2
    bias_update_rate: float = 0.0001

class MoERad(nn.Module):
    def __init__(self, config, dropout_p):
        super().__init__()
        self.config = config

        # Building Transformer
        self.transformer = nn.ModuleDict(
            dict(
                wpe_img = nn.Embedding(49, config.n_embd),
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = nn.Embedding(config.block_size, config.n_embd),
                h_e = nn.ModuleList([EncoderBlock(config, dropout_p) for _ in range(config.n_layer)]),
                h = nn.ModuleList([Block(config, dropout_p) for _ in range(config.n_layer)]),
                ln_f = nn.LayerNorm(config.n_embd)
            )
        )

        self.img_projection = nn.Linear(2048, config.n_embd, bias=False)

        # Building Linear layer
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight-sharing scheme
        self.transformer.wte.weight = self.lm_head.weight
        
        # Initialising Weights of transformer and linear head
        self.apply(self._init_weights)

    # Weight initialising function
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, img, idx, attn_mask=None, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        # print(idx)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb # (B, T, n_embd)
        
        # Image Processing --> img = (B, 1, 1024)
        pos_img = torch.arange(0, 49, dtype=torch.long, device=idx.device) # shape (49)
        pos_img_emb = self.transformer.wpe_img(pos_img) # position embeddings of shape (T, n_embd)

        img = self.img_projection(img)
        x_img = img + pos_img_emb.unsqueeze(0) #(B, 49, n_embd)

        for block in self.transformer.h_e:
            x_img = block(x_img)
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x, x_img, attn_mask)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)
        return logits, loss


class cap_dataset():
    def __init__(self):
        self.transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4981, 0.4981, 0.4981),
                                    (0.2288, 0.2288, 0.2288))])

        self.img_model = BCLModel(name='resnet50', num_classes=20, feat_dim=1024,
                                 use_norm=True)
        chkpt = torch.load('models/pretrained_BCL.pt')#, map_location='cpu')
        self.img_model.load_state_dict(chkpt)


    def preprocess_image(self, img_path):
        # Image
        img = Image.open(img_path).convert('RGB')
        img = self.transforms(img)

        self.img_model.eval()
        with torch.no_grad():
            feat_mlp, logits, _, img_patches = self.img_model(img.unsqueeze(0)) # (1, 1024), (1, 2048, 7, 7)   
        
        img_patches = img_patches.unsqueeze(0) #(1,1,2048,7,7)
        feat_mlp = feat_mlp.squeeze(0) # (1024)
        logits = logits.squeeze(0) # (20)
        x_img = img_patches.view(2048, 49).permute(1, 0) #(49, 2048)
        
        return x_img


def beam_search(model, img_features, start_token_id, end_token_id, 
                max_len=256, beam_width=3, num_beam_groups=3, diversity_penalty=1.0):
    device = img_features.device
    start_token = torch.tensor([start_token_id], dtype=torch.long).to(device)
    
    # Initialize beam groups
    beam_width_per_group = beam_width // num_beam_groups
    groups = [[(start_token.clone(), 0.0)] for _ in range(num_beam_groups)]

    for _ in range(max_len):
        new_groups = []
        all_prev_tokens = []
        
        for g_idx, group in enumerate(groups):
            new_beams = []
            
            for seq, cum_log_prob in group:
                if seq[-1].item() == end_token_id:
                    new_beams.append((seq, cum_log_prob))
                    continue

                # Forward pass
                with torch.no_grad():
                    # print(img_features.shape, seq.shape)
                    # print(img_features.unsqueeze(0).shape, seq.unsqueeze(0).shape)
                    # print(seq.unsqueeze(0))
                    logits, _ = model(img_features.unsqueeze(0), seq.unsqueeze(0))
                    log_probs = F.log_softmax(logits[:, -1, :], dim=-1).squeeze(0)
                    # print(log_probs.shape)

                # Apply diversity penalty to previous groups' tokens
                adjusted_probs = log_probs.clone()
                for token in all_prev_tokens:
                    adjusted_probs[token] -= diversity_penalty
                
                # Select top candidates for this group
                topk_probs, topk_indices = torch.topk(adjusted_probs, beam_width_per_group)
                # print(topk_indices, topk_probs)
                for i in range(beam_width_per_group):
                    # print(seq)
                    # raise KeyboardInterrupt
                    new_seq = torch.cat([seq, topk_indices[i].unsqueeze(0)])
                    new_log_prob = cum_log_prob + topk_probs[i].item()
                    new_beams.append((new_seq, new_log_prob))
            
            # Sort and prune beams for this group
            new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width_per_group]
            new_groups.append(new_beams)
            
            # Track generated tokens for diversity penalty
            all_prev_tokens.extend([seq[-1].item() for seq, _ in new_beams])

        groups = new_groups
        
        finished_counter = 0
        for b in range(len(new_beams)):
            if (new_beams[b][0] == 50256).sum().item() > 1:
                # print((beams[b][0] == 50256).sum().item())
                finished_counter += 1
        
        if finished_counter == len(new_beams):
            break

    # Select best sequence from all groups
    all_beams = [beam for group in groups for beam in group]
    best_seq = max(all_beams, key=lambda x: x[1])[0]
    return best_seq.tolist()