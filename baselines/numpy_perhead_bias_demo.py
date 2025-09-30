
import numpy as np
def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True); e = np.exp(x); return e/np.sum(e, axis=axis, keepdims=True)
def perhead_bias_scores(T, H, slopes):
    i=np.arange(T)[:,None]; j=np.arange(T)[None,:]; rel=np.maximum(0, i-j).astype(float); return -slopes[:,None,None]*rel[None,:,:]
def compare_mean_vs_perhead(T=16,H=4,slopes=None):
    if slopes is None: slopes=np.array([0.005,0.01,0.02,0.04], dtype=float)
    rng=np.random.default_rng(42); base=rng.normal(0,1,size=(H,T,T)); causal=np.triu(np.full((T,T), -np.inf), k=1)
    ph=perhead_bias_scores(T,H,slopes); scores_ph=base+ph+causal[None,:,:]; probs_ph=softmax(scores_ph, axis=-1)
    gmean=np.mean(slopes); bias_mean=-gmean*np.maximum(0, np.arange(T)[:,None]-np.arange(T)[None,:]).astype(float)
    probs_mean=softmax(base+bias_mean[None,:,:]+causal[None,:,:],axis=-1)
    eps=1e-12; kl=np.sum(probs_ph*(np.log(probs_ph+eps)-np.log(probs_mean+eps)),axis=-1)
    return np.mean(kl), kl, probs_ph, probs_mean
