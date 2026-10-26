"""Controlled 20-event pooling/GRU user towers with shared item initialization."""
from collections import deque
from itertools import groupby
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .data import write_json


class SequenceTower(nn.Module):
    def __init__(self,kind='pool',max_length=20):
        super().__init__()
        if kind not in ('pool','gru'):raise ValueError('Unknown user encoder')
        self.kind=kind;self.max_length=max_length
        # Construct this first in both models: same seed -> identical item init.
        self.item_tower=nn.Sequential(nn.Linear(250,128),nn.ReLU(),nn.Linear(128,64))
        if kind=='pool':self.user_tower=nn.Sequential(nn.Linear(250,128),nn.ReLU(),nn.Linear(128,64))
        else:self.user_tower=nn.GRU(250,64,batch_first=True)

    def items(self,vectors):return F.normalize(self.item_tower(vectors),dim=-1)

    def users(self,vectors):
        if vectors.ndim!=3 or vectors.shape[1]!=self.max_length or vectors.shape[2]!=250:
            raise ValueError('Expected a batch of fixed-length 250-D sequences')
        mask=vectors.abs().sum(dim=-1)>0
        lengths=mask.sum(dim=1)
        if not torch.all(lengths>0):
            raise ValueError('Expected nonempty right-padded sequence')
        # Right padding must be contiguous; hidden future slots are not allowed.
        positions=torch.arange(vectors.shape[1],device=vectors.device)[None,:]
        if not torch.equal(mask,positions<lengths[:,None]):raise ValueError('Noncontiguous padding')
        if self.kind=='pool':
            weights=.7**(lengths[:,None]-1-positions)
            pooled=(vectors*(weights*mask)[:,:,None]).sum(dim=1)
            result=self.user_tower(F.normalize(pooled,dim=-1))
        else:
            packed=nn.utils.rnn.pack_padded_sequence(vectors,lengths.cpu(),batch_first=True,enforce_sorted=False)
            _,hidden=self.user_tower(packed)
            result=hidden[-1]
        return F.normalize(result,dim=-1)

    def item_initialization_hash(self):
        digest=hashlib.sha256()
        for name,value in self.item_tower.state_dict().items():
            digest.update(name.encode());digest.update(value.detach().cpu().numpy().tobytes())
        return digest.hexdigest()


class SequenceProvider:
    def __init__(self,indices,store):self.indices=indices;self.store=store
    def __len__(self):return len(self.indices)
    def __getitem__(self,key):
        rows=np.asarray(self.indices[key])
        values=self.store.vectors(np.maximum(rows,0))
        return np.ascontiguousarray(np.where((rows>=0)[...,None],values,0),dtype=np.float32)


def build_sequences(connection,store,output,end,request_after=None,training=False,max_length=20):
    if max_length<1:raise ValueError('max_length must be positive')
    capacity=connection.execute('SELECT count(*) FROM clicks WHERE role="train" AND timestamp<=?',(end,)).fetchone()[0]
    if not capacity:raise ValueError('No source events')
    indices=np.lib.format.open_memmap(output/'indices.npy',mode='w+',dtype='int32',shape=(capacity,max_length))
    records=np.lib.format.open_memmap(output/'records.npy',mode='w+',dtype='int64',shape=(capacity,6))
    n=unavailable=tied=0
    cursor=connection.execute('SELECT user,article,timestamp,ordinal FROM clicks WHERE role="train" AND timestamp<=? '
                              'ORDER BY user,timestamp,ordinal',(end,))
    for user,events in groupby(cursor,lambda row:row[0]):
        history=deque(maxlen=max_length);pending=[];previous=None;previous_sequence=[]
        length=previous_length=0
        for _,article,timestamp,ordinal in events:
            row_index=store.lookup[article]
            if previous is not None:
                prev_time,prev_ordinal=previous
                if timestamp==prev_time:tied+=1
                elif request_after is None or prev_time>request_after:
                    if training and store.created[row_index]>prev_time:unavailable+=1
                    else:
                        indices[n]=-1
                        indices[n,:len(previous_sequence)]=previous_sequence
                        records[n]=(user,prev_ordinal,prev_time,row_index,timestamp,previous_length)
                        n+=1
            if previous is not None and timestamp!=previous[0]:
                history.extend(pending);length+=len(pending);pending=[]
            previous_sequence=(list(history)+[row_index])[-max_length:]
            previous_length=length+1
            pending.append(row_index);previous=(timestamp,ordinal)
    indices.flush();records.flush()
    info=dict(examples=n,capacity=capacity,max_length=max_length,training=training,request_after=request_after,end=end,
              unavailable_training_positives_excluded=unavailable,same_timestamp_transitions_in_source=tied,
              padding='right, vector row -1; current click plus strictly earlier timestamp batches')
    write_json(output/'examples.json',info)
    return indices[:n],records[:n],info
