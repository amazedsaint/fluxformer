
import torch
import torch.nn as nn
from .attn_ph import GaugeAttentionPH
from .rev_coupling import RevCouplingSegment

class FluxBlockV2(nn.Module):
    def __init__(self, d_model, pos_len, q_layers=3, use_attn=True, n_heads=8, attn_window=None,
                 use_topk_moe=False, n_experts=4, topk=2, ffn_mult=4):
        super().__init__()
        C = d_model + (d_model % 2)
        self.use_attn = use_attn
        self.ln_attn = nn.LayerNorm(C)
        self.attn = GaugeAttentionPH(d_model=C, n_heads=n_heads, dropdown=0.0, bias_g=0.02, window=attn_window)
        self.ln_ff = nn.LayerNorm(C)
        self.ff = RevCouplingSegment(C, hidden_mult=ffn_mult)
    def forward(self, x):
        h = self.ln_attn(x)
        if self.use_attn: x = x + self.attn(h)
        x = x + self.ff(self.ln_ff(x))
        return x

class FluxFormerV2(nn.Module):
    def __init__(self, vocab, d_model=768, n_layers=12, n_heads=12, pos_len=8192, q_layers=3,
                 attn_every=4, attn_window=512, use_topk_moe=False, n_experts=8, topk=2, ffn_mult=4):
        super().__init__()
        C = d_model + (d_model % 2)
        self.tok = nn.Embedding(vocab, C)
        blocks = []
        for i in range(n_layers):
            use_attn = (attn_every is None) or (i % attn_every == 0)
            blocks.append(FluxBlockV2(C, pos_len, q_layers=q_layers, use_attn=use_attn,
                                      n_heads=n_heads, attn_window=attn_window, ffn_mult=ffn_mult))
        self.blocks = nn.ModuleList(blocks)
        self.ln = nn.LayerNorm(C)
        self.head = nn.Linear(C, vocab)
    def forward(self, x):
        h = self.tok(x)
        for blk in self.blocks:
            h = blk(h)
        h = self.ln(h)
        return self.head(h)
