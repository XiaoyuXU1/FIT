import copy
import math
import time
import unittest
from types import SimpleNamespace

import torch
from transformers import BatchEncoding

from Fit.layer import importance,select_layers
from Unlearning_algorithm.gradient import train_request
from Evaluations.evaluator import evaluate


class Tokenizer:
    pad_token_id = 0

    def encode(self, text, **kwargs):
        return [1, 2] if text == "question" else [3, 4]

    def __call__(self, text, **kwargs):
        rows = [self.encode(x) for x in text] if isinstance(text, list) else [self.encode(text)]
        ids = torch.tensor(rows)
        return BatchEncoding({"input_ids": ids, "attention_mask": torch.ones_like(ids)})

    def batch_decode(self, tokens, **kwargs):
        return ["target answer" if row.tolist() == [3, 4] else "question target answer"
                for row in tokens]


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)])
        self.head = torch.nn.Linear(4, 8)
        self.config = SimpleNamespace(vocab_size=8)

    @property
    def device(self):
        return self.embedding.weight.device

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids=None, inputs_embeds=None, labels=None, **kwargs):
        x = self.embedding(input_ids) if inputs_embeds is None else inputs_embeds
        for layer in self.model.layers:
            x = x + layer(x)
        logits = self.head(x)
        loss = None if labels is None else torch.nn.functional.cross_entropy(
            logits[:, :-1].transpose(1, 2), labels[:, 1:])
        return SimpleNamespace(logits=logits, loss=loss)

    def generate(self, input_ids, **kwargs):
        return torch.cat([input_ids, torch.tensor([[3, 4]]).expand(len(input_ids), -1)], dim=1)


class UnlearningTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = TinyModel()
        self.tokenizer = Tokenizer()
        self.args = SimpleNamespace(chunk_size=4, epochs=2, lr=1e-3)

    def test_ga_increases_forget_loss(self):
        inputs=self.tokenizer('abcd')
        before=self.model(**inputs,labels=inputs.input_ids).loss.item()
        self.args.lr=.01
        train_request(self.model,None,self.tokenizer,'abcd',[], 'GA',self.args,math.inf)
        self.assertGreater(self.model(**inputs,labels=inputs.input_ids).loss.item(),before)

    def test_ga_retain_variants_keep_reference_fixed(self):
        for method in ['GA_GD','GA_KL']:
            model=copy.deepcopy(self.model)
            reference=copy.deepcopy(model).requires_grad_(False)
            before=copy.deepcopy(reference.state_dict())
            n=train_request(model,reference,self.tokenizer,'abcd',
                            [{'text':'retain'}],method,self.args,math.inf)
            self.assertEqual(n,2)
            self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
            self.assertTrue(all(torch.equal(p,before[k]) for k,p in reference.state_dict().items()))
            self.assertTrue(all(p.grad is None for p in reference.parameters()))

    def test_multichunk_npo_kl_updates_and_keeps_reference_fixed(self):
        reference = copy.deepcopy(self.model).requires_grad_(False)
        before = copy.deepcopy(reference.state_dict())
        n = train_request(self.model, reference, self.tokenizer, "abcdefgh",
                          [{"text": "retain"}], "NPO_KL", self.args, math.inf)
        self.assertEqual(n, 4)
        self.assertTrue(any(not torch.equal(p, before[k]) for k, p in self.model.state_dict().items()))
        self.assertTrue(all(torch.equal(p, before[k]) for k, p in reference.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_exhausted_budget_performs_no_updates(self):
        before = copy.deepcopy(self.model.state_dict())
        n = train_request(self.model, None, self.tokenizer, "abcd", [], "RL",
                          self.args, time.perf_counter()-1)
        self.assertEqual(n, 0)
        self.assertTrue(all(torch.equal(p, before[k]) for k, p in self.model.state_dict().items()))

    def test_attribution_restores_weights_and_importance_leaves_no_grads(self):
        before = copy.deepcopy(self.model.state_dict())
        self.assertEqual(len(select_layers(self.model, self.tokenizer, "data", 0.5)), 1)
        self.assertTrue(all(torch.equal(p, before[k]) for k, p in self.model.state_dict().items()))
        self.assertGreater(importance(self.model, self.tokenizer, "data"), 0)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_rouge_scores_only_generated_answer(self):
        result = evaluate(self.model, self.tokenizer,
                          [{"index": 0, "text": "text", "question": "question", "answer": "target answer"}])
        self.assertEqual(result["rougeL"], 1.0)

    def test_alkn_zero_step_size_preserves_weights_exactly(self):
        from Unlearning_algorithm.ALKN import ALKN
        before=copy.deepcopy(self.model.state_dict())
        self.args.lr=0.
        ALKN().run(self.model,self.tokenizer,'abcdefgh',[{'text':'retain'}],self.args,math.inf)
        self.assertTrue(all(torch.equal(p,before[k]) for k,p in self.model.state_dict().items()))

    def test_alkn_task_vector_negation_increases_target_loss(self):
        from Unlearning_algorithm.ALKN import ALKN
        inputs=self.tokenizer('abcdefgh')
        before=self.model(**inputs,labels=inputs.input_ids).loss.item()
        self.args.lr=.01
        ALKN().run(self.model,self.tokenizer,'abcdefgh',[{'text':'retain'}],self.args,math.inf)
        after=self.model(**inputs,labels=inputs.input_ids).loss.item()
        self.assertGreater(after,before)

    def test_o3_latest_adapter_matches_training_and_zero_gate_restores_base(self):
        from Unlearning_algorithm.O3 import OrthogonalLinear
        layer=OrthogonalLinear(torch.nn.Linear(4,4,bias=False)).eval()
        inputs=torch.randn(2,4)
        base=layer(inputs).detach()
        layer.add_adapter()
        with torch.no_grad(): layer.b[-1].fill_(.1)
        adapted=layer(inputs).detach()
        layer.finish(); layer.set_gate(1.)
        self.assertTrue(torch.allclose(adapted,layer(inputs),atol=1e-6))
        layer.set_gate(0.)
        self.assertTrue(torch.equal(base,layer(inputs)))

    def test_batched_filter_preserves_duplicate_and_empty_filter_decisions(self):
        from Fit.filtering import Data_filtering
        from Fit.filtering import BatchedFiltering
        class Original(Data_filtering):
            def __init__(self,epsilon):
                self.chunk_size=4; self.device='cpu'; self.similarity_threshold=.9; self.epsilon=epsilon
            def _get_embedding(self,text):
                return torch.tensor([[1.,0.]]) if text=='aaaa' else torch.tensor([[0.,1.]])
            def _compute_text_loss(self,text): return len(text)*.01
        class Batched(BatchedFiltering,Original):
            def __init__(self,epsilon): Original.__init__(self,epsilon)
            def _embeddings(self,chunks): return torch.cat([self._get_embedding(c) for c in chunks])
            def _losses(self,texts): return [self._compute_text_loss(t) for t in texts]
        history=[torch.tensor([1.,0.])]
        for text in ['aaaabbbb','aaaabbbbaaaa','aaaa','']:
            for epsilon in [.01,.2]:
                old=Original(epsilon).filter_data(text,history)
                new=Batched(epsilon).filter_data(text,history)
                self.assertEqual(old[:2],new[:2])
                self.assertEqual(len(old[2]),len(new[2]))

    def test_o3_batch_preserves_distinct_query_gates_and_bias(self):
        from Unlearning_algorithm.O3 import OrthogonalLinear
        layer=OrthogonalLinear(torch.nn.Linear(4,5,bias=True)).eval()
        for _ in range(2):
            layer.add_adapter()
            with torch.no_grad(): layer.b[-1].normal_()
            layer.finish()
        inputs=torch.randn(3,7,4)
        expected=[]
        for row,gate in zip(inputs,[0.,.4,1.]):
            layer.set_gate(gate)
            expected.append(layer(row))
        layer.batch_gates=torch.tensor([0.,.4,1.])
        self.assertTrue(torch.allclose(layer(inputs),torch.stack(expected),atol=2e-6))
        layer.set_gate(0.)
        self.assertTrue(torch.equal(layer(inputs),layer.base(inputs)))

    def test_o3_resumes_latest_adapter_and_does_not_sum_task_deltas(self):
        from Unlearning_algorithm.O3 import OrthogonalLinear
        layer=OrthogonalLinear(torch.nn.Linear(4,5,bias=False)).eval()
        layer.add_adapter()
        with torch.no_grad(): layer.b[-1].fill_(.1)
        layer.finish()
        previous_a=layer.a[-1].detach().clone()
        previous_b=layer.b[-1].detach().clone()
        inputs=torch.randn(3,4)
        previous_delta=2*torch.nn.functional.linear(torch.nn.functional.linear(inputs,previous_a),previous_b)
        layer.add_adapter()
        self.assertTrue(torch.equal(layer.a[-1],previous_a))
        self.assertTrue(torch.equal(layer.b[-1],previous_b))
        with torch.no_grad():
            layer.a[-1].mul_(.7)
            layer.b[-1].add_(.2)
        expected=2*torch.nn.functional.linear(torch.nn.functional.linear(inputs,layer.a[-1]),layer.b[-1])
        layer.finish()
        actual=layer(inputs)-layer.base(inputs)
        self.assertTrue(torch.allclose(actual,expected,atol=1e-6))
        self.assertFalse(torch.allclose(actual,previous_delta+expected,atol=1e-6))
        self.assertTrue(torch.equal(layer.a[0],previous_a))

    def test_o3_scienceqa_gate_remapping(self):
        import numpy as np
        from scipy.special import ndtr,expit
        from Unlearning_algorithm.O3 import inference_weights
        x=np.array([0.,1.,1.5,2.,100.])
        first,second=ndtr(x),ndtr(-x)
        raw=expit(10*(1-np.maximum(first,second)+np.minimum(first,second))-2)
        expected=np.array([1.2,0.,raw[2],0.,0.])
        self.assertTrue(np.array_equal(inference_weights(first,second),expected))

    def test_o3_query_batch_cached_decoding_matches_individual_queries(self):
        from transformers import LlamaConfig,LlamaForCausalLM
        from Unlearning_algorithm.O3 import OrthogonalLinear
        model=LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=16,
            intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,
            num_key_value_heads=2,pad_token_id=0,eos_token_id=2)).eval()
        layers=[]
        for name,module in list(model.named_modules()):
            if isinstance(module,torch.nn.Linear) and '.layers.' in name:
                parent,attr=name.rsplit('.',1)
                layer=OrthogonalLinear(module).eval()
                setattr(model.get_submodule(parent),attr,layer)
                layer.add_adapter()
                with torch.no_grad(): layer.b[-1].normal_(std=.02)
                layer.finish(); layers.append(layer)
        ids=torch.tensor([[0,0,3,4],[5,6,7,8]])
        expected=[]
        for row,gate in zip(ids,[0.,.7]):
            for layer in layers: layer.set_gate(gate)
            expected.append(model.generate(input_ids=row[None],attention_mask=row.ne(0)[None],
                max_new_tokens=4,min_new_tokens=4,do_sample=False,use_cache=True))
        for layer in layers:
            layer.batch_gates=torch.tensor([0.,.7])
        actual=model.generate(input_ids=ids,attention_mask=ids.ne(0),max_new_tokens=4,
                              min_new_tokens=4,do_sample=False,use_cache=True)
        self.assertTrue(torch.equal(actual,torch.cat(expected)))

    def test_verbose_answer_still_counts_as_target_leakage(self):
        from Evaluations.evaluator import exact_answer,token_f1
        answer='Liam Ember'
        generated='Liam Ember '+('additional unrelated continuation '*30)
        self.assertLess(token_f1(answer,generated),.8)
        self.assertTrue(exact_answer(answer,generated))

    def test_curate_runner_writes_requests_without_auxiliary_files(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from Unlearning_algorithm import CURaTE as run_curate
        rows=[{'index':i,'text':f'text {i}','question':f'question {i}',
               'answer':'answer'} for i in range(3)]
        embedder=SimpleNamespace(encode=lambda texts,**kw:torch.tensor([[1.,0.]]))
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'CURaTE'
            args=SimpleNamespace(model="test",embedder="test",output=str(output),
                max_requests=2,eval_limit=1,eval_every=60,seed=42,device="cpu")
            with (patch.object(run_curate,'pch_records',return_value=rows),
                  patch.object(run_curate,'load_model',return_value=self.model),
                  patch.object(run_curate,'load_tokenizer',return_value=self.tokenizer),
                  patch.object(run_curate,'SentenceTransformer',return_value=embedder),
                  patch.object(run_curate,'generate_answers',side_effect=lambda m,t,qs:['answer']*len(qs)),
                  patch.object(run_curate,'stage_responses',return_value=({},[])),
                  patch.object(run_curate,'measure_latency',return_value={}),
                  patch.object(torch.cuda,'synchronize'),
                  patch.object(torch.cuda,'max_memory_allocated',return_value=0)):
                run_curate.run(args)
            requests=[json.loads(s) for s in (output/'requests.jsonl').read_text().splitlines()]
            self.assertEqual([r['database_entries'] for r in requests],[1,2])
            self.assertEqual(json.loads((output/'completed.json').read_text())['requests'],2)

    def test_answer_match_rejects_part_of_a_longer_word_or_number(self):
        from Evaluations.evaluator import exact_answer
        for answer,response,expected in [('62','The value is 16285680.',False),
                                         ('Ann','Annual report',False),
                                         ('62','The value is 62.',True),
                                         ('Liam Ember','Liam Ember: more details.',True)]:
            self.assertEqual(exact_answer(answer,response),expected)


if __name__ == "__main__":
    unittest.main()
