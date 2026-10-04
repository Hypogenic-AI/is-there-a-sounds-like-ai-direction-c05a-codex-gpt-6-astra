from common import *
from huggingface_hub import snapshot_download
for m,rev in read('models.json').items():
 print(m,flush=True)
 snapshot_download(m,revision=rev,allow_patterns=['*.json','*.safetensors','pytorch_model.bin','*.txt','*.model','merges.txt','vocab.json'],max_workers=4)
