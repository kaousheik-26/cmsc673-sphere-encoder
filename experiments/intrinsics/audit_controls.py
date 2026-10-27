"""Extra front-facing normal and random/individual unseen-depth controls."""
import argparse
import importlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from .property_metrics import normal_metrics
from .depth_feasibility import similarity
from .shared_transform import SharedTransform
from .train_shared_depth import decode_with_grad


# Check simple normal baselines, random depth edits, and target-aware per-image depth optimization.
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--loader-module',default='experiments.intrinsics.traverse')
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    import torch.distributed as dist
    dist.init_process_group('gloo',init_method=(a.output_dir.resolve()/'dist-init').as_uri(),rank=0,world_size=1)
    try:
        result={'normal_constant_controls':{},'depth_random_unseen':[],'depth_per_image_unseen':[]}
        normals=torch.load(a.root/'results-property-normals-01/pairs.pt',map_location='cpu',weights_only=True)
        train=[i for i,r in enumerate(normals['manifest']) if r['split']=='train']
        average=F.normalize((normals['target'][train]*2-1).mean((0,2,3)),dim=0)
        fields={'front':torch.tensor([0.5,0.5,1.]),'training_mean_vector':(average+1)/2}
        for split in ['test','unseen']:
            indices=[i for i,r in enumerate(normals['manifest']) if r['split']==split]
            for name,vector in fields.items():
                records=[normal_metrics(vector[:,None,None].expand_as(normals['target'][i]),normals['target'][i]) for i in indices]
                result['normal_constant_controls'][f'{name}_{split}']={key:sum(r[key] for r in records)/len(records) for key in records[0]}
        del normals
        data=torch.load(a.root/'results-property-depth-01/pairs.pt',map_location='cpu',weights_only=True)
        indices=[i for i,r in enumerate(data['manifest']) if r['split']=='unseen']
        loader=importlib.import_module(a.loader_module)
        model=loader.build_checkpoint_model(json.loads((a.checkpoint.parent.parent/'cfg.json').read_text()),a.checkpoint,'ema',True).cuda()
        for method in ['geodesic','addition']:
            for seed in [673,674,675]:
                state=torch.load(a.root/f'results-shared-01/{method}_seed{seed}.pt',map_location='cpu',weights_only=True)['state_dict']
                torch.manual_seed(seed+1000)
                edit=SharedTransform(torch.randn_like(state['direction']).cuda(),method,model.spherify_mode).cuda()
                edit.raw_size.data.copy_(state['raw_size'])
                with torch.no_grad():
                    records=[similarity(decode_with_grad(model,edit(data['z'][i:i+1].cuda()))[0],data['target'][i].cuda()) for i in indices]
                result['depth_random_unseen'].append({'method':method,'seed':seed,'rgb_mse':sum(r['rgb_mse'] for r in records)/len(records),'gray_pearson':sum(r['gray_pearson'] for r in records)/len(records)})
        for index in indices[::6][:4]:
            edit=SharedTransform((data['target_z'][index:index+1]-data['z'][index:index+1]).cuda(),'geodesic',model.spherify_mode).cuda()
            opt=torch.optim.Adam([{'params':[edit.direction],'lr':0.03},{'params':[edit.raw_size],'lr':0.01}])
            for step in range(150):
                opt.zero_grad(set_to_none=True)
                loss=(decode_with_grad(model,edit(data['z'][index:index+1].cuda()))-data['target'][index:index+1].cuda()).square().mean()
                loss.backward();torch.nn.utils.clip_grad_norm_(edit.parameters(),1.);opt.step()
            with torch.no_grad():
                prediction=decode_with_grad(model,edit(data['z'][index:index+1].cuda()))[0]
                metrics=similarity(prediction,data['target'][index].cuda());metrics['index']=index
                result['depth_per_image_unseen'].append(metrics)
                loader.save_grid([[data['rgb'][index].cuda(),data['target'][index].cuda(),prediction]],[['RGB','Depth teacher','Per-image optimized']],a.output_dir/f'depth_reference_{index}.png')
        result['frozen_verified']=not any(x.requires_grad or x.grad is not None for x in model.parameters())
        (a.output_dir/'results.json').write_text(json.dumps(result,indent=2))
        print('Additional controls completed',flush=True)
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
