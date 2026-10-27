"""Categorical-loss control for fixed-palette shared segmentation editing."""
import argparse
import copy
import importlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from .shared_transform import SharedTransform
from .train_shared_depth import decode_with_grad
from .property_metrics import segmentation_palette, confusion_matrix, segmentation_metrics


def palette_logits(images, palette, temperature=0.02):
    """Differentiable negative squared distance to every fixed palette color."""
    pixels = images.permute(0,2,3,1)
    distance = pixels.square().sum(-1,keepdim=True) - 2 * pixels @ palette.T + palette.square().sum(1)
    return (-distance / temperature).permute(0,3,1,2)


# Fit shared edits with class-aware loss to test whether RGB matching caused segmentation failures.
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--pairs',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--loader-module',default='experiments.intrinsics.traverse')
    p.add_argument('--steps',type=int,default=400)
    a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4); torch.manual_seed(673)
    import torch.distributed as dist
    dist.init_process_group('gloo',init_method=(a.output_dir.resolve()/'dist-init').as_uri(),rank=0,world_size=1)
    try:
        loader=importlib.import_module(a.loader_module)
        model=loader.build_checkpoint_model(json.loads((a.checkpoint.parent.parent/'cfg.json').read_text()),a.checkpoint,'ema',True).cuda()
        data=torch.load(a.pairs,map_location='cpu',weights_only=True)
        splits={s:[i for i,r in enumerate(data['manifest']) if r['split']==s] for s in ['train','validation','test','unseen']}
        palette=segmentation_palette().cuda()
        initial=(data['target_z'][splits['train']]-data['z'][splits['train']]).mean(0,keepdim=True).cuda()
        # Pool class confusion counts and save teacher/output comparisons for a shared edit.
        @torch.no_grad()
        def evaluate(edit,indices,tag=None):
            total=torch.zeros(150,150,dtype=torch.long,device='cuda'); records=[]; rows=[]; captions=[]
            for position,index in enumerate(indices):
                prediction=decode_with_grad(model,edit(data['z'][index:index+1].cuda()))[0]
                truth=data['labels'][index].cuda()
                cm=confusion_matrix(prediction,truth,palette); total+=cm
                rec=segmentation_metrics(cm);rec['index']=index; records.append(rec)
                if tag and position<12:
                    row=[data['rgb'][index].cuda(),data['target'][index].cuda(),prediction]
                    caption=['RGB','Teacher segmentation',tag]
                    rows.append(row);captions.append(caption)
                    loader.save_grid([row],[caption],a.output_dir/f'{tag}_{index:03d}.png')
            if tag: loader.save_grid(rows,captions,a.output_dir/f'{tag}_overview.png')
            return {'summary':segmentation_metrics(total),'per_image':records}
        results={'objective':'cross entropy of palette-distance logits at 64px plus 0.1*RGB MSE','temperature':0.02,'selection':'validation pooled mIoU','steps':a.steps,'runs':[]}
        for method in ['geodesic','addition']:
            for seed in [673,674,675]:
                torch.manual_seed(seed)
                edit=SharedTransform(initial+0.05*torch.randn_like(initial),method,model.spherify_mode).cuda()
                opt=torch.optim.Adam([{'params':[edit.direction],'lr':0.03},{'params':[edit.raw_size],'lr':0.01}])
                best,state,beststep=-1.,None,0; curve=[];rng=torch.Generator().manual_seed(seed)
                for step in range(a.steps+1):
                    if step%50==0 or step==a.steps:
                        val=evaluate(edit,splits['validation'])['summary']['miou']
                        curve.append({'step':step,'validation_miou':val})
                        if val>best:best,state,beststep=val,copy.deepcopy(edit.state_dict()),step
                        print(f'categorical {method} seed={seed} step={step} val_mIoU={val:.6f}',flush=True)
                    if step==a.steps:break
                    ids=torch.tensor(splits['train'])[torch.randint(len(splits['train']),(4,),generator=rng)]
                    opt.zero_grad(set_to_none=True)
                    prediction=decode_with_grad(model,edit(data['z'][ids].cuda()))
                    resized=F.interpolate(prediction,size=(64,64),mode='area')
                    labels=F.interpolate(data['labels'][ids,None].float().cuda(),size=(64,64),mode='nearest')[:,0].long()
                    loss=F.cross_entropy(palette_logits(resized,palette),labels)+0.1*(prediction-data['target'][ids].cuda()).square().mean()
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite categorical loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(edit.parameters(),1.);opt.step()
                edit.load_state_dict(state)
                torch.save({'state_dict':state,'method':method,'seed':seed,'beststep':beststep},a.output_dir/f'{method}_seed{seed}.pt')
                run={'method':method,'seed':seed,'beststep':beststep,'curve':curve,'validation_miou':best,
                     'test':evaluate(edit,splits['test'],tag=f'{method}_test' if seed==673 else None),
                     'unseen':evaluate(edit,splits['unseen'],tag=f'{method}_unseen' if seed==673 else None)}
                results['runs'].append(run)
                (a.output_dir/'results.json').write_text(json.dumps(results,indent=2))
        # Four target-informed, per-image categorical references distinguish
        # the sharing constraint from the RGB-only optimization objective.
        refs=[]
        for index in splits['unseen'][::6][:4]:
            edit=SharedTransform((data['target_z'][index:index+1]-data['z'][index:index+1]).cuda(),'geodesic',model.spherify_mode).cuda()
            opt=torch.optim.Adam([{'params':[edit.direction],'lr':0.03},{'params':[edit.raw_size],'lr':0.01}])
            labels=F.interpolate(data['labels'][index:index+1,None].float().cuda(),size=(64,64),mode='nearest')[:,0].long()
            for step in range(150):
                opt.zero_grad(set_to_none=True)
                prediction=decode_with_grad(model,edit(data['z'][index:index+1].cuda()))
                loss=F.cross_entropy(palette_logits(F.interpolate(prediction,size=(64,64),mode='area'),palette),labels)
                loss.backward();torch.nn.utils.clip_grad_norm_(edit.parameters(),1.);opt.step()
            refs.append(evaluate(edit,[index],tag=f'per_image_{index}'))
        results['per_image_references']=refs
        results['frozen_verified']=not any(p.requires_grad or p.grad is not None for p in model.parameters())
        if not results['frozen_verified']:raise RuntimeError('Model not frozen')
        (a.output_dir/'results.json').write_text(json.dumps(results,indent=2))
        print('Categorical segmentation control completed',flush=True)
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
