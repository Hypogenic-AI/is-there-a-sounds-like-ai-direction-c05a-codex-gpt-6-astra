"""Residual extraction, frozen readout fits and controlled autoregressive interventions.
Post-block hooks operate in native coordinates before final normalization.
Inspired by CAA mean differences and assistant-axis last-position hooks; implemented
locally to make padding, cache and positional choices explicit.
"""
from common import *
import argparse, gc, platform
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer,AutoModelForCausalLM

torch.set_num_threads(8)
LAYERS=[8,16,24]
def unit(v):return v/max(np.linalg.norm(v),1e-12)
def load_model(kind):
 name='Qwen/Qwen2.5-1.5B'+('-Instruct' if kind=='instruct' else '')
 rev=read('models.json')[name]
 tok=AutoTokenizer.from_pretrained(name,revision=rev,padding_side='left')
 tok.pad_token=tok.eos_token
 model=AutoModelForCausalLM.from_pretrained(name,revision=rev,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
 return tok,model

@torch.inference_mode()
def extract(tok,model,texts,layers=LAYERS,batch=32):
 """Last nonpadding token at each exact post-block hook, and response NLL."""
 saved={l:[] for l in layers}; hs={}; handles=[]
 for l in layers:
  def hook(mod,inp,out,l=l):
   z=out[0] if isinstance(out,tuple) else out
   hs[l]=z[:,-1].float().cpu().numpy()
  handles.append(model.model.layers[l-1].register_forward_hook(hook))
 nll=[];length=[]
 try:
  for start in range(0,len(texts),batch):
   b=tok(texts[start:start+batch],padding=True,truncation=True,max_length=192,return_tensors='pt').to('cuda')
   o=model(**b,use_cache=False)
   for l in layers:saved[l].append(hs[l])
   # Next-token NLL excludes padding and transitions from padding to first token.
   loss=torch.nn.functional.cross_entropy(o.logits[:,:-1].float().transpose(1,2),b.input_ids[:,1:],reduction='none')
   mask=b.attention_mask[:,1:]*b.attention_mask[:,:-1]
   nll.extend(((loss*mask).sum(1)/mask.sum(1).clamp_min(1)).cpu().tolist());length.extend(b.attention_mask.sum(1).cpu().tolist())
 finally:
  for h in handles:h.remove()
 return {l:np.concatenate(v) for l,v in saved.items()},np.array(nll),np.array(length)

class Intervention:
 """Add or ablate at last prompt token and every cached generation token."""
 def __init__(self,model,layer,v,amount=0,ablate=False):
  self.model=model;self.layer=layer;self.v=torch.as_tensor(v,device='cuda',dtype=torch.float32);self.amount=amount;self.ablate=ablate
 def hook(self,mod,inp,out):
  x=out[0] if isinstance(out,tuple) else out
  y=x.clone();last=x[:,-1].float()
  delta=-(last@self.v)[:,None]*self.v if self.ablate else self.amount*self.v
  y[:,-1]=(last+delta).to(x.dtype)
  return (y,)+out[1:] if isinstance(out,tuple) else y
 def __enter__(self):self.handle=self.model.model.layers[self.layer-1].register_forward_hook(self.hook);return self
 def __exit__(self,*args):self.handle.remove()

def format_prompt(tok,prompt,kind):
 if kind=='instruct':return tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True)
 return 'Instruction: '+prompt+'\nResponse:'

@torch.inference_mode()
def generate(tok,model,prompts,kind,seednum=42,sample=True,max_new=160):
 seed(seednum)
 b=tok([format_prompt(tok,p,kind) for p in prompts],padding=True,return_tensors='pt').to('cuda')
 kw=dict(max_new_tokens=max_new,do_sample=sample,pad_token_id=tok.pad_token_id,use_cache=True)
 if sample:kw.update(temperature=.7,top_p=.9,top_k=0)
 out=model.generate(**b,**kw)[:,b.input_ids.shape[1]:]
 texts=tok.batch_decode(out,skip_special_tokens=True)
 counts=[int((x!=tok.pad_token_id).sum()) for x in out]
 trunc=[not bool((x==tok.eos_token_id).any()) for x in out]
 return texts,counts,trunc

def nuisance_texts(tok,model,kind):
 """Separate invented source tasks; contrasts hold supplied facts constant."""
 path=f'nuisance_{kind}.json'
 if (R/path).exists():return read(path)
 result=[]
 for axis,positive,negative in [('formality','Use formal professional prose.','Use casual conversational prose.'),('verbosity','Give a detailed 100-word explanation.','Be concise, using about 30 words.'),('assistant','You are a helpful AI assistant explaining this information to a user.','You are a fictional local narrator recounting this event to a friend.')]:
  for sign,instruction in [(1,positive),(-1,negative)]:
   prompts=[]
   for j in range(24):
    fact=f'The River{j} community center opens a pottery course on June {j+1}. It has {12+j} seats. The teacher is Alex. The fee is {20+j} dollars.'
    prompts.append(instruction+' Preserve these facts and output only the text: '+fact)
   for start in range(0,24,16):
    txt,_,_=generate(tok,model,prompts[start:start+16],kind,sample=False,max_new=128)
    result.extend(dict(axis=axis,sign=sign,id=start+i,prompt=prompts[start+i],text=x) for i,x in enumerate(txt))
 save(path,result);return result

def run(kind):
 start=time.time();seed(42);tok,model=load_model(kind);corpus=read('corpus.json');df=pd.DataFrame(corpus)
 cache=R/f'activations_{kind}.npz'
 if cache.exists():
  data=np.load(cache);acts={l:data[f'h{l}'] for l in LAYERS};nll=data['nll'];length=data['length']
 else:
  acts,nll,length=extract(tok,model,df.text.tolist());np.savez_compressed(cache,**{f'h{l}':x for l,x in acts.items()},nll=nll,length=length)
 tr=(df.split=='train').values;va=(df.split=='val').values;y=df.label.values
 vectors={};metrics=[]
 for l,x in acts.items():
  v=unit(x[tr&(y==1)].mean(0)-x[tr&(y==0)].mean(0));clf=LogisticRegression(C=1,max_iter=2000).fit(x[tr],y[tr]);w=unit(clf.coef_[0]);vectors[l]=(v,w)
  for split in ['val','test','ood','hc3']:
   sub=df[df.split==split]
   for gen in sorted(set(sub.generator)-{'human'}):
    mask=((df.split==split)&df.generator.isin(['human',gen])).values
    for method,vec in [('mean',v),('logistic',w)]:metrics.append(dict(model=kind,layer=l,split=split,generator=gen,method=method,auc=roc_auc_score(y[mask],x[mask]@vec),n=int(mask.sum())))
 gpt=read('data_audit.json')['training_generator'];val=[r for r in metrics if r['split']=='val' and r['method']=='mean'];chosen=max(val,key=lambda r:r['auc'])['layer'];v,w=vectors[chosen];x=acts[chosen];scale=float(np.std(x[tr]@v));print(kind,'selected',chosen,'scale',scale,flush=True)
 nuis=nuisance_texts(tok,model,kind);na,_,_=extract(tok,model,[r['text'] for r in nuis],layers=[chosen]);nx=na[chosen];nd=pd.DataFrame(nuis)
 dirs={'mean':v,'logistic':w}
 for axis in ['formality','verbosity','assistant']:
  dirs[axis]=unit(nx[((nd.axis==axis)&(nd.sign==1)).values].mean(0)-nx[((nd.axis==axis)&(nd.sign==-1)).values].mean(0))
 # Length covariance and an academic-vs-other-domain contrast use training only.
 dirs['length']=unit((x[tr]-x[tr].mean(0)).T@(length[tr]-length[tr].mean()))
 dom=(df.domain=='acad').values;dirs['domain']=unit(x[tr&dom].mean(0)-x[tr&~dom].mean(0))
 basis=np.stack([dirs[k] for k in ['formality','verbosity','assistant','length','domain']],axis=1);q,_=np.linalg.qr(basis);res=v-q@(q.T@v);dirs['residualized']=unit(res)
 rng=np.random.default_rng(198);dirs.update({f'random{i}':unit(rng.normal(size=len(v))) for i in range(5)})
 np.savez_compressed(R/f'vectors_{kind}.npz',**dirs,scale=scale,layer=chosen)
 geom={'model':kind,'layer':chosen,'scale':scale,'val_auc':max(r['auc'] for r in val),'cosines':{k:float(v@vv) for k,vv in dirs.items()},'residual_norm':float(np.linalg.norm(res)),'added_norm_over_mean_residual_norm_at_2':2*scale/float(np.linalg.norm(x[tr],axis=1).mean()),'projection_length_spearman':float(spearmanr(x[tr]@v,length[tr]).statistic),'projection_nll_spearman':float(spearmanr(x[tr]@v,nll[tr]).statistic)}
 save(f'geometry_{kind}.json',geom);save(f'readout_{kind}.json',metrics)
 # Validate native hook arithmetic, zero equivalence, and deterministic repetition.
 fake=torch.randn(2,3,len(v),device='cuda',dtype=torch.bfloat16);hook=Intervention(model,chosen,v,2*scale);actual=hook.hook(None,None,fake)
 expected=(fake[:,-1].float()+2*scale*torch.tensor(v,device='cuda')).bfloat16();assert torch.equal(actual[:,-1],expected) and torch.equal(actual[:,:-1],fake[:,:-1])
 test=['Write one sentence about a community garden.']*2
 a=generate(tok,model,test,kind,sample=False,max_new=20)[0]
 with Intervention(model,chosen,v,0):b=generate(tok,model,test,kind,sample=False,max_new=20)[0]
 assert a==b
 c=generate(tok,model,test,kind,sample=True,max_new=20)[0];d=generate(tok,model,test,kind,sample=True,max_new=20)[0];assert c==d
 save(f'hook_validation_{kind}.json',{'zero_equivalent':a==b,'repeat_equal':c==d,'arithmetic':True,'example':c})
 state(f'Phase 3 implemented and numerically validated for {kind}: extraction/readout, native hooks, nuisance fit and reproducible sampling. Phase 4 active: generation with frozen controls. Layer selection only used held-out readout validation. Artifacts results/*_{kind}.*. Next independent encoder/API scoring, then paired analysis and report.')
 tasks=read('tasks.json');conditions=[('zero','mean',0,False),('mean_m1','mean',-1,False),('mean_p1','mean',1,False),('mean_m2','mean',-2,False),('mean_p2','mean',2,False),('logistic_m2','logistic',-2,False),('ablate','mean',0,True),*[(f'random{i}',f'random{i}',-2,False) for i in range(5)],('formality_m2','formality',-2,False),('assistant_m2','assistant',-2,False),('residual_m2','residualized',-2,False),('residual_p2','residualized',2,False),('human_prompt','mean',0,False)]
 if kind=='base':conditions=[c for c in conditions if c[0] in ['zero','mean_m2','mean_p2']]
 done={(r['model'],r['condition'],r['seed'],r['id']) for r in lines('generations.jsonl')}
 for sd in ([42,43] if kind=='instruct' else [42]):
  for condition,k,alpha,ablate in conditions:
   if sd==43 and condition not in ['zero','mean_m2','mean_p2']:continue
   for offset in range(0,len(tasks),16):
    batch=tasks[offset:offset+16]
    if all((kind,condition,sd,t['id']) in done for t in batch):continue
    prompts=[t['prompt']+(' Write like a human, using natural everyday language.' if condition=='human_prompt' else '') for t in batch]
    with Intervention(model,chosen,dirs[k],alpha*scale,ablate):texts,counts,trunc=generate(tok,model,prompts,kind,seednum=sd+offset)
    for t,text,n,cut in zip(batch,texts,counts,trunc):
     append('generations.jsonl',dict(model=kind,condition=condition,seed=sd,id=t['id'],domain=t['domain'],source=t['source'],prompt=prompts[batch.index(t)],text=text,tokens=n,truncated=cut,anchor_recall=np.mean([a.lower() in text.lower() for a in t['anchors']]),layer=chosen,alpha=alpha))
   print(kind,sd,condition,'done',round(time.time()-start,1),flush=True)
 save(f'runtime_{kind}.json',{'seconds':time.time()-start,'gpu':torch.cuda.get_device_name(),'peak_bytes':torch.cuda.max_memory_allocated(),'torch':torch.__version__,'python':platform.python_version(),'dtype':'bfloat16','activation_batch':32,'generation_batch':16})
 del model;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--model',choices=['instruct','base'],default='instruct');args=p.parse_args();run(args.model)
