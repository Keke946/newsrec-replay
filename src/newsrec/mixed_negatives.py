"""Budget-matched negative-source intervention, not unbiased sampled softmax."""
import hashlib
import numpy as np
import torch
from torch.nn import functional as F
from .two_tower import negative_mask


def state_hash(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def mixed_mask(targets,request_times,created,catalog,rng):
    targets=np.asarray(targets);times=np.asarray(request_times)
    original=negative_mask(torch.from_numpy(targets.copy()),torch.from_numpy(created[targets].copy()),torch.from_numpy(times.copy())).numpy()
    off=original.copy();np.fill_diagonal(off,False)
    counts=off.sum(axis=1);needed=counts//2;keep=counts-needed
    retained=np.eye(len(targets),dtype=bool)
    priorities=rng.random(off.shape);priorities[~off]=np.inf
    ordered=np.argsort(priorities,axis=1)
    for i,k in enumerate(keep):retained[i,ordered[i,:k]]=True
    size=max(256,4*int(needed.max(initial=0)))
    for _ in range(12):
        pool=rng.choice(catalog,size=size,replace=True)
        eligible=(created[pool][None,:]<=times[:,None])&(pool[None,:]!=targets[:,None])
        if np.all(eligible.sum(axis=1)>=needed):break
        size*=2
    else:raise ValueError('Could not draw enough legal uniform negatives')
    selected=eligible&(eligible.cumsum(axis=1)<=needed[:,None])
    mask=np.concatenate((retained,selected),axis=1)
    if not np.array_equal(mask.sum(axis=1)-1,counts):raise ValueError('Actual negative budget differs')
    return pool,mask,counts


def train_mixed(model,store,queries,records,end,batch_size=256,seed=2026):
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    order=np.random.default_rng(seed).permutation(len(records));rng=np.random.default_rng(seed+1)
    catalog=np.flatnonzero(store.created<=end)
    model.train();loss_sum=count=updates=negative_count=0
    for start in range(0,len(order),batch_size):
        chosen=order[start:start+batch_size]
        if len(chosen)<2:continue
        rows=np.asarray(records[chosen]);targets=rows[:,3]
        pool,mask,negative_counts=mixed_mask(targets,rows[:,2],store.created,catalog,rng)
        if not negative_counts.sum():continue
        query=torch.from_numpy(np.array(queries[chosen],copy=True))
        candidates=torch.from_numpy(store.vectors(np.concatenate((targets,pool))))
        logits=model.users(query)@model.items(candidates).T/.1
        logits=logits.masked_fill(~torch.from_numpy(mask),-torch.inf)
        loss=F.cross_entropy(logits,torch.arange(len(chosen)))
        if not torch.isfinite(loss):raise ValueError('Nonfinite mixed-negative loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step();count+=len(chosen);updates+=1
        loss_sum+=float(loss.detach())*len(chosen);negative_count+=int(negative_counts.sum())
    return dict(updates=updates,examples=count,mean_loss=loss_sum/count if count else None,usable_negative_logits=negative_count,
                warning='same actual negative count per request, but different source and encoding cost; no logQ correction')
