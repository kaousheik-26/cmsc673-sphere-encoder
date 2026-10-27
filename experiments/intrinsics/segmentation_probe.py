"""Frozen-latent linear semantic probe; diagnostic, not a decoder edit."""
import argparse
import copy
import json
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from .property_metrics import segmentation_metrics, segmentation_palette


# Train a small classifier on cached frozen latents to test accessible semantic information.
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--steps',type=int,default=1000)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    data=torch.load(a.pairs,map_location='cpu',weights_only=True)
    z=data['z']; labels=data['labels']; classes=150
    splits={s:[i for i,r in enumerate(data['manifest']) if r['split']==s] for s in ['train','validation','test','unseen']}
    side=int(z.shape[1]**0.5)
    assert side**2==z.shape[1]
    features=z.transpose(1,2).reshape(len(z),z.shape[2],side,side)
    from PIL import Image
    import numpy as np
    palette=segmentation_palette()
    # Upsample classifier predictions and compare them with teacher labels; the decoder is not used.
    @torch.no_grad()
    def evaluate(probe,indices,tag=None):
        matrix=torch.zeros(classes,classes,dtype=torch.long,device='cuda')
        records=[]
        for pos,index in enumerate(indices):
            logits=F.interpolate(probe(features[index:index+1].cuda()),size=(256,256),mode='bilinear',align_corners=False)
            prediction=logits.argmax(1)[0]
            truth=labels[index].cuda()
            cm=torch.bincount((truth.flatten()*classes+prediction.flatten()),minlength=classes**2).reshape(classes,classes)
            matrix+=cm; record=segmentation_metrics(cm); record['index']=index; records.append(record)
            if tag and pos<12:
                target_image=(palette[labels[index]].numpy()*255).round().astype(np.uint8)
                probe_image=(palette[prediction.cpu()].numpy()*255).round().astype(np.uint8)
                original=(data['rgb'][index].permute(1,2,0).numpy()*255).round().astype(np.uint8)
                canvas=Image.new('RGB',(768,256))
                for col,array in enumerate([original,target_image,probe_image]):canvas.paste(Image.fromarray(array),(col*256,0))
                canvas.save(a.output_dir/f'{tag}_{index:03d}.png')
        return {'summary':segmentation_metrics(matrix),'per_image':records}
    results={'diagnostic':'Linear 1x1 token classifier on cached frozen RGB latents, no decoder editing','steps':a.steps,'runs':[]}
    for seed in [673,674,675]:
        torch.manual_seed(seed)
        probe=nn.Conv2d(z.shape[2],classes,1).cuda()
        optimizer=torch.optim.AdamW(probe.parameters(),lr=0.003,weight_decay=1e-4)
        best,state,beststep=-1.,None,0; curve=[];rng=torch.Generator().manual_seed(seed)
        for step in range(a.steps+1):
            if step%100==0 or step==a.steps:
                score=evaluate(probe,splits['validation'])['summary']['miou']
                curve.append({'step':step,'validation_miou':score})
                if score>best:best,state,beststep=score,copy.deepcopy(probe.state_dict()),step
                print(f'probe seed={seed} step={step} val_mIoU={score:.6f}',flush=True)
            if step==a.steps:break
            ids=torch.tensor(splits['train'])[torch.randint(len(splits['train']),(4,),generator=rng)]
            optimizer.zero_grad(set_to_none=True)
            logits=F.interpolate(probe(features[ids].cuda()),size=(64,64),mode='bilinear',align_corners=False)
            truth=F.interpolate(labels[ids,None].float().cuda(),size=(64,64),mode='nearest')[:,0].long()
            loss=F.cross_entropy(logits,truth)
            loss.backward();optimizer.step()
        probe.load_state_dict(state)
        torch.save({'state_dict':state,'seed':seed,'beststep':beststep},a.output_dir/f'probe_seed{seed}.pt')
        results['runs'].append({'seed':seed,'beststep':beststep,'curve':curve,'test':evaluate(probe,splits['test'],tag='test' if seed==673 else None),'unseen':evaluate(probe,splits['unseen'],tag='unseen' if seed==673 else None)})
        (a.output_dir/'results.json').write_text(json.dumps(results,indent=2))
    print('Segmentation probe completed',flush=True)


if __name__=='__main__':main()
