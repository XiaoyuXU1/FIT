"""PCH adaptation of ALKN Algorithm 1 / equations 6--10.

The author's linked repository is unavailable (404). This is a paper-based
reimplementation with explicit task-vector negation.
State tensors are offloaded to CPU; the retain gradient uses a fixed minibatch.
"""
import math
import time

import torch
import torch.nn.functional as F

from utils import encode


class ALKN:
    def __init__(self):
        self.previous_mask = {}
        self.quantile_indices = {}
        self.sampling_generator = torch.Generator().manual_seed(42)
        self.start_weights = {}
        self.retain_gradient = {}
        self.vectors = {}
        self.gradients = {}
        self.staging = {}
        self.current_mask = {}

    def _cpu_gradient(self,name,gradient,storage):
        if name not in storage:
            storage[name]=torch.empty_like(gradient,device='cpu',dtype=torch.float32)
        dtype=gradient.dtype
        if dtype not in self.staging or self.staging[dtype].numel()<gradient.numel():
            self.staging[dtype]=torch.empty(gradient.numel(),device='cpu',dtype=dtype)
        # Reuse the D2H staging allocation as well as the FP32 destination;
        # preserve the original BF16-to-FP32 conversion on CPU.
        staging=self.staging[dtype][:gradient.numel()].view(gradient.shape)
        staging.copy_(gradient.detach())
        storage[name].copy_(staging)
        return storage[name]

    def _sample_mask_vector(self,vectors):
        if not self.quantile_indices:
            total=sum(v.numel() for v in vectors.values())
            if total<=131072:
                self.quantile_indices={n:torch.arange(v.numel()) for n,v in vectors.items()}
            else:
                draws=torch.randint(total,(131072,),generator=self.sampling_generator)
                offset=0
                for name,value in vectors.items():
                    selected=draws[(draws>=offset)&(draws<offset+value.numel())]-offset
                    if selected.numel(): self.quantile_indices[name]=selected
                    offset+=value.numel()
        return torch.cat([vectors[n].flatten()[i] for n,i in self.quantile_indices.items()])

    def run(self, model, tokenizer, text, retain, args, deadline):
        parameters = dict(model.named_parameters())
        for name,p in parameters.items():
            if name not in self.start_weights:
                self.start_weights[name]=torch.empty_like(p,device='cpu')
            self.start_weights[name].copy_(p.detach())
        start_weights=self.start_weights
        chunks = [text[i:i+args.chunk_size] for i in range(0,len(text),args.chunk_size)]
        retain_gradient, vectors = self.retain_gradient,self.vectors
        moments, initial_probs = {}, []
        model.train()
        model.zero_grad(set_to_none=True)
        # A fixed retain minibatch avoids repeatedly scanning the entire benchmark
        # and has the same retain availability as the other PCH methods.
        inputs = encode(tokenizer, retain[0]["text"][:args.chunk_size], model.device,args.chunk_size)
        model(**inputs, labels=inputs.input_ids).loss.backward()
        for name, p in parameters.items():
            if p.grad is not None:
                self._cpu_gradient(name,p.grad,retain_gradient)
                if name not in vectors:
                    vectors[name]=torch.empty_like(retain_gradient[name])
                vectors[name].zero_()
        model.zero_grad(set_to_none=True)
        model.eval()
        with torch.no_grad():
            for chunk in chunks:
                x = encode(tokenizer,chunk,model.device,args.chunk_size)
                if x is None:
                    initial_probs.append(None)
                    continue
                logits=model(**x).logits[:,:-1].float()
                log_probs=F.log_softmax(logits,-1).gather(-1,x.input_ids[:,1:,None]).squeeze(-1)
                initial_probs.append(log_probs.exp().cpu())
        model.train()
        step,total,final_masks=0,max(1,args.epochs*len(chunks)),{}
        try:
            for epoch in range(args.epochs):
                for index,chunk in enumerate(chunks):
                    if time.perf_counter() >= deadline:
                        break
                    if initial_probs[index] is None:
                        continue
                    model.zero_grad(set_to_none=True)
                    x=encode(tokenizer,chunk,model.device,args.chunk_size)
                    logits=model(**x).logits[:,:-1].float()
                    logp=F.log_softmax(logits,-1).gather(-1,x.input_ids[:,1:,None]).squeeze(-1)
                    soft_label=torch.sigmoid(10*(initial_probs[index].to(model.device)-1))
                    loss=-(soft_label*logp).mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError("ALKN ESL loss is not finite")
                    loss.backward()
                    step+=1
                    warmup=int(.1*total)
                    scheduler_step=step-1
                    scale=scheduler_step/max(1,warmup) if scheduler_step<warmup else .1+.45*(1+math.cos(math.pi*(scheduler_step-warmup)/max(1,total-warmup)))
                    lr=args.lr*scale
                    with torch.no_grad():
                        gradients={}
                        count,gradient_sum,gradient_square_sum=0,0.,0.
                        for name,p in parameters.items():
                            if p.grad is None or name not in vectors:
                                continue
                            g_cpu=self._cpu_gradient(name,p.grad,self.gradients)
                            gradients[name]=g_cpu
                            variance,average=torch.var_mean(g_cpu,unbiased=False)
                            n=g_cpu.numel(); avg=average.item()
                            count+=n
                            gradient_sum+=n*avg
                            gradient_square_sum+=n*(variance.item()+avg*avg)
                        mu=math.sqrt(max(0.,gradient_square_sum/count-(gradient_sum/count)**2))
                        for name,g_cpu in gradients.items():
                            previous=self.previous_mask.get(name)
                            m=vectors[name]
                            m.addcmul_(g_cpu,retain_gradient[name],value=.8)
                            if previous is not None:
                                m.add_(previous,alpha=-mu)
                        # Estimate the paper's *global* robust quantiles with a
                        # uniform fixed sample over all parameters, not separate
                        # thresholds that would activate mostly-zero embeddings.
                        sample=self._sample_mask_vector(vectors)
                        k=max(1,int(sample.numel()*.15))
                        lo=sample.kthvalue(k).values
                        hi=sample.kthvalue(max(1,sample.numel()-k+1)).values
                        active_ratio=1-.7*((step-1)/max(1,total-1))
                        threshold=(lo+(hi-lo)*((1-active_ratio)-.15)/.7).item()
                        for name in gradients:
                            p=parameters[name];g=p.grad.detach()
                            previous=self.previous_mask.get(name)
                            if name not in self.current_mask:
                                self.current_mask[name]=torch.empty_like(vectors[name],dtype=torch.bool)
                            mask=self.current_mask[name]
                            if step==1:
                                mask.fill_(True)
                            else:
                                torch.ge(vectors[name],threshold,out=mask)
                            final_masks[name]=mask
                            device_mask=mask.to(p.device)
                            if name not in moments:
                                moments[name]=(torch.zeros_like(p),torch.zeros_like(p))
                            avg,var=moments[name]
                            masked_gradient=g*device_mask
                            avg.mul_(.9).add_(masked_gradient,alpha=.1)
                            var.mul_(.95).addcmul_(masked_gradient,masked_gradient,value=.05)
                            delta=avg/(1-.9**step)/(var.sqrt()/math.sqrt(1-.95**step)+1e-8)
                            delta.mul_(device_mask)
                            if previous is not None:
                                delta.mul_(torch.where(previous.to(p.device),.1,1.0))
                            p.mul_(1-lr*.01).add_(delta,alpha=-lr)
                        gradients.clear()
                    del logits,logp,loss
                if time.perf_counter() >= deadline:
                    break
        finally:
            # Negate the task vector even when a time budget ends mid-task.
            model.zero_grad(set_to_none=True)
            moments.clear()
            with torch.no_grad():
                for name,p in parameters.items():
                    old=start_weights[name].to(p.device)
                    reflected=p.float().sub_(old.float()).mul_(-.8).add_(old.float())
                    p.copy_(reflected)
            model.zero_grad(set_to_none=True)
        for name,mask in final_masks.items():
            if name in self.previous_mask:
                self.previous_mask[name].logical_or_(mask)
            else:
                self.previous_mask[name]=mask.clone()
        return step
