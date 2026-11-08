"""Architecture-independent PCH port of O³'s adapters and official OOD model.

Source: GCYZSL/O3-LLM-UNLEARNING @ 839afea7fc09001433ef30bd6670a6d6804a9045.
PCH request paragraphs are split into overlapping 128-character OOD examples.
This adaptation is recorded separately from the author's original benchmarks.
"""
import importlib.util
import hashlib
import urllib.request
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import ndtr, expit
from sklearn.svm import OneClassSVM
from sklearn.covariance import EmpiricalCovariance
from threadpoolctl import threadpool_limits
from transformers import AutoTokenizer

from utils import encode


class OrthogonalLinear(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.a = torch.nn.ParameterList()
        self.b = torch.nn.ParameterList()
        self.batch_gates = None
        self.training_adapter = None
        self.gate = 1.
        self.dropout = torch.nn.Dropout(.05)

    def add_adapter(self):
        for p in list(self.a)+list(self.b): p.requires_grad_(False)
        a = torch.empty(8,self.base.in_features,device=self.base.weight.device,dtype=self.base.weight.dtype)
        torch.nn.init.kaiming_uniform_(a,a=math.sqrt(5))
        b = torch.zeros(self.base.out_features,8,device=a.device,dtype=a.dtype)
        if len(self.a):
            # Official stage scripts reload the last adapter checkpoint, while
            # retaining earlier A snapshots only for the orthogonality penalty.
            with torch.no_grad():
                a.copy_(self.a[-1])
                b.copy_(self.b[-1])
            self.b[-1]=torch.nn.Parameter(self.b[-1].detach().cpu(),requires_grad=False)
        self.a.append(torch.nn.Parameter(a))
        self.b.append(torch.nn.Parameter(b))
        self.training_adapter=len(self.a)-1
        self.batch_gates=None

    def forward(self, x):
        if self.training_adapter is not None:
            i=self.training_adapter
            return self.base(x)+2*F.linear(F.linear(self.dropout(x),self.a[i]),self.b[i])
        base=self.base(x)
        if not len(self.a): return base
        adapted=2*F.linear(F.linear(x,self.a[-1]),self.b[-1])
        if self.batch_gates is not None:
            if x.shape[0]!=self.batch_gates.numel():
                raise ValueError('O³ query gate count differs from the generation batch')
            gates=self.batch_gates.view(-1,*([1]*(x.ndim-1)))
            return base+(gates*adapted.float()).to(base.dtype)
        return base+(self.gate*adapted.float()).to(base.dtype)

    def orthogonal_loss(self):
        if len(self.a)<2: return self.a[-1].sum()*0
        previous=torch.cat([p.detach() for p in self.a[:-1]],dim=0)
        return (previous.float()@self.a[-1].float().T).square().sum()

    @torch.no_grad()
    def finish(self):
        self.a[-1].requires_grad_(False)
        self.b[-1].requires_grad_(False)
        self.training_adapter=None

    @torch.no_grad()
    def set_gate(self, value):
        self.batch_gates=None
        self.gate=value


def inference_weights(first, second):
    raw=expit(10*(1-np.maximum(first,second)+np.minimum(first,second))-2)
    # Released ScienceQA eval_o3.py inference profile, without PCH tuning.
    return np.where(raw>.9,1.2,np.where((raw>.3)&(raw<=.4),raw,0.))


def detector_source():
    target=Path(__file__).resolve().parents[1]/'.cache/o3-source/ood_model_selector.py'
    revision='839afea7fc09001433ef30bd6670a6d6804a9045'
    url=f'https://raw.githubusercontent.com/GCYZSL/O3-LLM-UNLEARNING/{revision}/ScienceQA_expriment/src/ood_model_selector.py'
    data=target.read_bytes() if target.exists() else urllib.request.urlopen(url,timeout=60).read()
    if hashlib.sha256(data).hexdigest()!='08ed1cf87eb3f9d4d8737a2c6d5c6f5da27c94e4aae98d16ae248a7409ce65fd':
        raise ValueError('O³ author detector checksum mismatch')
    if not target.exists():
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(data)
    return target


class O3:
    def __init__(self, model, device):
        self.model,self.device=model,device
        model.requires_grad_(False)
        self.layers=[]
        for name,module in list(model.named_modules()):
            if isinstance(module,torch.nn.Linear) and '.layers.' in name:
                parent_name,attr=name.rsplit('.',1)
                parent=model.get_submodule(parent_name)
                wrapped=OrthogonalLinear(module)
                setattr(parent,attr,wrapped)
                self.layers.append(wrapped)
        if not self.layers: raise ValueError('No decoder linear modules found for O³')
        source=detector_source()
        spec=importlib.util.spec_from_file_location('o3_author_detector',source)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.detector=module.RobertaForSelector('FacebookAI/roberta-base',projection_dim=100).to(device)
        # Transformers 5 no longer consistently propagates the old load-time
        # output flag through PEFT. Make the author's required 13 states explicit.
        def hidden_states_hook(module, positional, keywords):
            keywords['output_hidden_states']=True
            keywords['return_dict']=True
            return positional,keywords
        self.detector.roberta.register_forward_pre_hook(hidden_states_hook,with_kwargs=True)
        self.detector.roberta_k.register_forward_pre_hook(hidden_states_hook,with_kwargs=True)
        self.detector_tokenizer=AutoTokenizer.from_pretrained('FacebookAI/roberta-base')
        self.initial={n:p.detach().cpu().clone() for n,p in self.detector.named_parameters() if p.requires_grad}
        self.calibrations=[]
        self.query_cache={}
        model.prepare_queries=self.prepare_queries
        model.prepare_query=self.prepare_query
        model.prepare_query_batch=self.prepare_query_batch

    def _restore(self, state):
        params=dict(self.detector.named_parameters())
        with torch.no_grad():
            for n,value in state.items(): params[n].copy_(value)

    @torch.no_grad()
    def _features(self, texts):
        self.detector.eval()
        batches=[]
        for start in range(0,len(texts),32):
            x=self.detector_tokenizer(texts[start:start+32],max_length=128,truncation=True,
                                      padding='max_length',return_tensors='pt').to(self.device)
            out=self.detector.roberta(**x)
            batches.append(torch.stack([h.float().mean(1) for h in out.hidden_states],0))
        return torch.cat(batches,dim=1)

    def _score(self, features, calibration):
        scores=[]
        for i in range(1,13):
            diff=features[i]-calibration['mean'][i]
            precision=calibration['precision'][i].to(features.device)
            maha=-.5*((diff@precision)@diff.T).diag()
            cosine=(F.normalize(features[i],dim=-1)@calibration['bank'][i].T).amax(-1)
            scores.append(maha-1000*cosine)
        return torch.stack(scores,-1).cpu().numpy()

    def _train_detector(self, text):
        pieces=[text[i:i+128] for i in range(0,max(1,len(text)-63),64) if text[i:i+128].strip()]
        if len(pieces)<4:
            pieces=[text[i:i+64] for i in range(0,max(1,len(text)-31),32) if text[i:i+64].strip()]
        if len(pieces)<3:
            raise ValueError('PCH OOD training requires at least three nonempty windows')
        split=max(2,int(.8*len(pieces)))
        train,test=pieces[:split],pieces[split:]
        if not test: test=train[-2:]
        self._restore(self.initial)
        self.detector.train()
        params=[p for p in self.detector.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=1e-5,eps=1e-6,weight_decay=.01)
        for _ in range(2):
            for start in range(0,len(train),8):
                x=self.detector_tokenizer(train[start:start+8],padding='max_length',truncation=True,
                                          max_length=128,return_tensors='pt').to(self.device)
                masked={k:v.clone() for k,v in x.items()}
                chosen=torch.rand_like(x.input_ids.float())<.15
                special=torch.zeros_like(chosen)
                for token in self.detector_tokenizer.all_special_ids:
                    special |= x.input_ids.eq(token)
                chosen &= ~special
                if not chosen.any(): chosen[0,1]=True
                masked['labels']=x.input_ids.masked_fill(~chosen,-100)
                # BERT-style 80% mask, 10% random, 10% unchanged MLM corruption.
                draw=torch.rand_like(x.input_ids.float())
                masked['input_ids'][chosen & (draw<.8)]=self.detector_tokenizer.mask_token_id
                random_mask=chosen & (draw>=.8) & (draw<.9)
                random_ids=torch.randint(len(self.detector_tokenizer),x.input_ids.shape,device=self.device)
                masked['input_ids'][random_mask]=random_ids[random_mask]
                optimizer.zero_grad(set_to_none=True)
                mlm,entropy=self.detector(batch_mlm=masked,batch=x)
                loss=mlm+entropy
                if not torch.isfinite(loss): raise FloatingPointError('O³ detector loss')
                loss.backward()
                optimizer.step()
        del optimizer
        features=self._features(train)
        means,precision,banks=[],[],[]
        with threadpool_limits(limits=8):
            for layer in features:
                cpu_layer=layer.cpu()
                mean=cpu_layer.mean(0)
                centered=(cpu_layer-mean).numpy()
                estimator=EmpiricalCovariance(assume_centered=False).fit(centered)
                precision.append(torch.from_numpy(estimator.precision_).float())
                means.append(mean.to(self.device))
                banks.append(F.normalize(cpu_layer,dim=-1).to(self.device))
        calibration={'mean':means,'precision':precision,'bank':banks,
                     'state':{n:p.detach().cpu().clone() for n,p in self.detector.named_parameters() if p.requires_grad}}
        train_features=self._score(features,calibration)
        test_features=self._score(self._features(test),calibration)
        classifier=OneClassSVM(nu=.1,kernel='linear').fit(train_features)
        scores_train=classifier.score_samples(train_features)
        scores_test=classifier.score_samples(test_features)
        calibration.update(classifier=classifier,means=(float(scores_train.mean()),float(scores_test.mean())),
                           std=max(float(scores_test.std()),1e-6))
        calibration['precision']=[value.cpu() for value in precision]
        self.calibrations.append(calibration)

    def run(self, tokenizer, text, args, deadline):
        if time.perf_counter()>=deadline: return 0
        self._train_detector(text)
        if time.perf_counter()>=deadline:
            self.calibrations.pop()
            return 0
        for layer in self.layers: layer.add_adapter()
        params=[p for layer in self.layers for p in (layer.a[-1],layer.b[-1])]
        optimizer=torch.optim.AdamW(params,lr=3e-4,weight_decay=0.)
        chunks=[text[i:i+args.chunk_size] for i in range(0,len(text),args.chunk_size)]
        total_steps=15*len(chunks)
        warmup=min(100,int(.1*total_steps))
        def lr_scale(step):
            if step<warmup: return step/max(1,warmup)
            return max(0.,(total_steps-step)/max(1,total_steps-warmup))
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_scale)
        self.model.train()
        steps=0
        for _ in range(15):
            for chunk in chunks:
                if time.perf_counter()>=deadline: break
                x=encode(tokenizer,chunk,self.device,args.chunk_size)
                if x is None: continue
                labels=torch.randint(self.model.config.vocab_size,x.input_ids.shape,device=self.device)
                optimizer.zero_grad(set_to_none=True)
                loss=self.model(**x,labels=labels).loss
                if not torch.isfinite(loss): raise FloatingPointError('O³ LoRA loss')
                loss.backward()
                for layer in self.layers:
                    orth=.1*layer.orthogonal_loss()/len(self.model.model.layers)
                    if not torch.isfinite(orth): raise FloatingPointError('O³ orthogonality loss')
                    orth.backward()
                torch.nn.utils.clip_grad_norm_(params,1.)
                optimizer.step()
                scheduler.step()
                steps+=1
            if time.perf_counter()>=deadline: break
        del optimizer
        for layer in self.layers: layer.finish()
        self.model.zero_grad(set_to_none=True)
        return steps

    @torch.no_grad()
    def prepare_queries(self, queries):
        for index,calibration in enumerate(self.calibrations):
            needed=list(dict.fromkeys(q for q in queries if self.query_cache.get(q,(0.,0))[1]<=index))
            if not needed: continue
            self._restore(calibration['state'])
            features=self._features(needed)
            scores=calibration['classifier'].score_samples(self._score(features,calibration))
            a,b=calibration['means']
            std=calibration['std']
            center=(a+b)/2
            cdf=lambda x:.5*ndtr((x-a)/std)+.5*ndtr((x-b)/std)
            first,second=cdf(scores),cdf(2*center-scores)
            weights=inference_weights(first,second)
            for q,w in zip(needed,weights):
                old=self.query_cache.get(q,(0.,0))[0]
                self.query_cache[q]=(max(old,float(w)),index+1)

    @torch.no_grad()
    def prepare_query(self, query):
        self.prepare_queries([query])
        gate=self.query_cache.get(query,(0.,0))[0]
        for layer in self.layers: layer.set_gate(gate)
        return gate

    @torch.no_grad()
    def prepare_query_batch(self,queries):
        self.prepare_queries(queries)
        gates=torch.tensor([self.query_cache.get(q,(0.,0))[0] for q in queries],
                           device=self.device,dtype=torch.float32)
        for layer in self.layers:
            layer.batch_gates=gates
