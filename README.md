# DGCRE Source code


> Continual relation extraction aims to continuously learn new relation categories without forgetting the already learned ones. To achieve this goal, two key issues need to be addressed: catastrophic forgetting (CF) of the model and knowledge transfer (KT) of the relations. In terms of CF, there has been a great deal of research work. However, another important challenge of continual learning: knowledge transfer, has hardly been studied in the field of relation extraction. To address this, we propose dynamically constructing relation extraction networks (DCREN) for Continual relation extraction, which dynamically changes the architecture of the model through six designed actions to achieve knowledge transfer of similar relations, and further to combat catastrophic forgetting, an extensible classification module is proposed to expand the new learning space for new tasks while preserving the knowledge of old relations. Experiments show that DCREN achieves state-of-the-art performance through dynamically updating the model structure to learn new relations and transfer old knowledge.

## Environment
Our implementation is based on Python 3.9 and the version of PyTorch is 1.9.1 (cuda version 12.x).  
To install PyTorch 1.9.1, you could follow the official guidance of [PyTorch](https://pytorch.org/).  



Then, other dependencies could be installed by running:
```
pip install -r requirements.txt
```

## Dataset
We use two datasets in our experiments, FewRel and TACRED.

The splited datasets and task orders could be found in the corresponding directory of `data/`.




We conduct all experiments on a single RTX 4090 GPU with 24GB memory.

