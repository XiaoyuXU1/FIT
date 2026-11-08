"""Batch the existing filtering rule without changing its keep/delete decisions."""
import torch
import torch.nn.functional as F

from Fit.filtering import Data_filtering


class BatchedFiltering(Data_filtering):
    @torch.no_grad()
    def _embeddings(self,chunks):
        vectors=[]
        for start in range(0,len(chunks),128):
            inputs=self.tokenizer(chunks[start:start+128],return_tensors='pt',truncation=True,
                                  max_length=self.chunk_size,padding=True).to(self.device)
            vectors.append(self.embedding_model(**inputs,return_dict=True).pooler_output)
        return torch.cat(vectors,0)

    @torch.no_grad()
    def _losses(self,texts):
        values=[0.]*len(texts)
        pending=[i for i,t in enumerate(texts) if t.strip()]
        for start in range(0,len(pending),4):
            indices=pending[start:start+4]
            x=self.language_tokenizer([texts[i] for i in indices],padding=True,truncation=True,
                                      max_length=4096,return_tensors='pt').to(self.device)
            logits=self.language_model(**x,use_cache=False).logits[:,:-1].float()
            labels=x.input_ids[:,1:]
            mask=x.attention_mask[:,1:]*x.attention_mask[:,:-1]
            ce=F.cross_entropy(logits.transpose(1,2),labels,reduction='none')
            losses=(ce*mask).sum(1)/mask.sum(1)
            for i,value in zip(indices,losses.tolist()): values[i]=value
        return values

    def filter_data(self,new_data,unlearning_memory):
        text=new_data.strip()
        if not text: return '','',[]
        chunks=self._chunk_text(text)
        embeddings=self._embeddings(chunks)
        if not unlearning_memory: return text,'',list(embeddings)
        history=torch.stack(unlearning_memory).to(self.device)
        similarity=(F.normalize(embeddings,dim=-1,eps=1e-8)@
                    F.normalize(history,dim=-1,eps=1e-8).T).max(-1).values
        candidates=[i for i,s in enumerate(similarity.tolist()) if s>=self.similarity_threshold
                    and not torch.all(embeddings[i]==0)]
        deleted=set()
        if candidates:
            # Each candidate removes all equal chunks, exactly as in the original.
            unique=list(dict.fromkeys(chunks[i] for i in candidates))
            losses=self._losses([''.join(chunks)]+[''.join(c for c in chunks if c!=target) for target in unique])
            decisions={c:abs(losses[0]-loss)<=self.epsilon for c,loss in zip(unique,losses[1:])}
            deleted={i for i in candidates if decisions[chunks[i]]}
        kept=[i for i in range(len(chunks)) if i not in deleted]
        return ''.join(chunks[i] for i in kept),''.join(chunks[i] for i in range(len(chunks)) if i in deleted),[embeddings[i] for i in kept]
