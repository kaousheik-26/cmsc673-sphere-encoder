import argparse
import copy
import hashlib
import importlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .geometry import normalize
from .depth_feasibility import depth_to_target, similarity
from .property_metrics import normal_metrics, segmentation_palette, confusion_matrix, segmentation_metrics
from .shared_transform import SharedTransform
from .train_shared_depth import decode_with_grad


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--previous-results', type=Path, required=True)
    p.add_argument('--unseen-manifest', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--cache-dir', type=Path, required=True)
    p.add_argument('--loader-module', default='experiments.intrinsics.traverse')
    p.add_argument('--property', choices=['depth', 'normals', 'segmentation'], required=True)
    p.add_argument('--steps', type=int, default=400)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(673)
    import torch.distributed as dist
    import numpy as np
    from PIL import Image
    from huggingface_hub import HfApi
    from sphere.loader import center_crop_arr
    loader = importlib.import_module(args.loader_module)
    dist.init_process_group('gloo', init_method=(args.output_dir.resolve() / 'dist-init').as_uri(), rank=0, world_size=1)
    try:
        cfg = json.loads((args.checkpoint.parent.parent / 'cfg.json').read_text())
        model = loader.build_checkpoint_model(cfg, args.checkpoint, 'ema', True).cuda()
        old = torch.load(args.previous_results / 'pairs.pt', map_location='cpu', weights_only=True)
        unseen = json.loads(args.unseen_manifest.read_text())
        manifest = copy.deepcopy(old['manifest']) + unseen
        splits = {s: [i for i,r in enumerate(manifest) if r['split']==s] for s in ['train', 'validation', 'test', 'unseen']}
        if len(set(r['image'] for r in manifest)) != len(manifest):
            raise ValueError('Overlapping image paths')
        old_categories = {r['category'] for r in old['manifest']}
        if old_categories & {r['category'] for r in unseen}:
            raise ValueError('Unseen categories overlap training categories')
        palette = segmentation_palette().cuda()
        teacher_id = {'depth':'depth-anything/Depth-Anything-V2-Small-hf',
                      'normals':'prs-eth/marigold-normals-v1-1',
                      'segmentation':'nvidia/segformer-b0-finetuned-ade-512-512'}[args.property]
        revision = '5426e4f0f36572d16453bbda7a8389317b1bef99' if args.property == 'depth' else HfApi().model_info(teacher_id).sha
        if args.property == 'normals':
            from diffusers import MarigoldNormalsPipeline
            teacher = MarigoldNormalsPipeline.from_pretrained(teacher_id, variant='fp16', torch_dtype=torch.float16, cache_dir=str(args.cache_dir), revision=revision).to('cuda')
            teacher.set_progress_bar_config(disable=True)
            for component in [teacher.unet, teacher.vae, teacher.text_encoder]:
                component.eval().requires_grad_(False)
        else:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation, AutoModelForSemanticSegmentation
            processor = AutoImageProcessor.from_pretrained(teacher_id, cache_dir=str(args.cache_dir), revision=revision)
            cls = AutoModelForDepthEstimation if args.property == 'depth' else AutoModelForSemanticSegmentation
            teacher = cls.from_pretrained(teacher_id, cache_dir=str(args.cache_dir), revision=revision).eval().requires_grad_(False).cuda()
        print(f'{args.property} teacher loaded at {revision}', flush=True)
        rgb, targets, zs, ztargets, labels = [], [], [], [], []
        hashes = set()
        with torch.no_grad():
            for i, row in enumerate(manifest):
                with Image.open(row['image']) as image:
                    cropped = center_crop_arr(image.convert('RGB'), 256)
                array = np.array(cropped)
                digest = hashlib.sha256(array.tobytes()).hexdigest()
                if digest in hashes:
                    raise ValueError(f'Duplicate crop at {row["image"]}')
                hashes.add(digest)
                row['cropped_rgb_sha256'] = digest
                original = torch.from_numpy(array.copy()).permute(2,0,1).float().cuda() / 255
                if args.property == 'depth' and i < len(old['manifest']):
                    target = old['target'][i].cuda()
                elif args.property == 'normals':
                    prediction = teacher(cropped, num_inference_steps=4, ensemble_size=1, processing_resolution=512,
                                         generator=torch.Generator(device='cuda').manual_seed(673+i)).prediction
                    normal = torch.from_numpy(prediction[0].copy()).permute(2,0,1).float().cuda()
                    if normal.shape != (3,256,256) or not torch.isfinite(normal).all():
                        raise ValueError('Invalid normal target')
                    target = (F.normalize(normal, dim=0) + 1) / 2
                else:
                    prediction = teacher(**processor(images=cropped, return_tensors='pt').to('cuda'))
                    if args.property == 'depth':
                        raw = processor.post_process_depth_estimation(prediction, target_sizes=[(256,256)])[0]['predicted_depth']
                        target, bounds = depth_to_target(raw)
                        row['depth_normalization'] = bounds
                    else:
                        label = processor.post_process_semantic_segmentation(prediction, target_sizes=[(256,256)])[0]
                        labels.append(label.cpu())
                        target = palette[label].permute(2,0,1)
                rgb.append(original.cpu()); targets.append(target.cpu())
                zs.append(normalize(model.f(model.encode(original[None]*2-1)).float(), model.spherify_mode)[0].cpu())
                ztargets.append(normalize(model.f(model.encode(target[None]*2-1)).float(), model.spherify_mode)[0].cpu())
                if (i+1)%12 == 0:
                    print(f'{args.property}: prepared {i+1}/{len(manifest)}', flush=True)
        del teacher
        torch.cuda.empty_cache()
        rgb, targets, zs, ztargets = map(torch.stack, [rgb, targets, zs, ztargets])
        labels = torch.stack(labels) if labels else None
        torch.save({'rgb':rgb, 'target':targets, 'z':zs, 'target_z':ztargets, 'labels':labels, 'manifest':manifest}, args.output_dir/'pairs.pt')
        (args.output_dir/'manifest.json').write_text(json.dumps(manifest, indent=2))
        template = targets[splits['train']].mean(0).cuda()
        if labels is not None:
            majority_id = torch.bincount(labels[splits['train']].flatten(), minlength=150).argmax().item()
            majority = palette[majority_id][:,None,None].expand(3,256,256)

        @torch.no_grad()
        def evaluate(edit, indices, tag=None, fixed=None, target_latent=False):
            records, matrix = [], torch.zeros(150,150, device='cuda', dtype=torch.long)
            rows, text = [], []
            for position,index in enumerate(indices):
                source = zs[index:index+1].cuda()
                moved = ztargets[index:index+1].cuda() if target_latent else edit(source) if edit is not None else source
                prediction = fixed if fixed is not None else decode_with_grad(model,moved)[0]
                truth = targets[index].cuda()
                record = similarity(prediction,truth)
                record['index'] = index
                record['shuffled_mse'] = (prediction-targets[indices[(position+1)%len(indices)]].cuda()).square().mean().item()
                record['radius_error'] = loader.latent_stats(source,moved,model.spherify_mode)['max_relative_radius_error']
                if args.property == 'normals':
                    record.update(normal_metrics(prediction,truth))
                elif labels is not None:
                    cm = confusion_matrix(prediction,labels[index].cuda(),palette)
                    matrix += cm
                    record.update(segmentation_metrics(cm))
                records.append(record)
                if tag and position < 12:
                    row = [rgb[index].cuda(), truth, prediction, template, decode_with_grad(model,ztargets[index:index+1].cuda())[0]]
                    caption = ['RGB','Teacher target',tag,'Training template','Target reconstruction']
                    loader.save_grid([row],[caption],args.output_dir/f'{tag}_{index:03d}.png')
                    rows.append(row); text.append(caption)
            keys = ['rgb_mse','gray_pearson','shuffled_mse','radius_error']
            if args.property == 'normals': keys += ['mean_angle_deg','fraction_under_11_25_deg']
            summary = {k:sum(r[k] for r in records if r[k] is not None)/sum(r[k] is not None for r in records) for k in keys if any(r[k] is not None for r in records)}
            if labels is not None: summary.update(segmentation_metrics(matrix))
            if tag: loader.save_grid(rows,text,args.output_dir/f'{tag}_overview.png')
            return {'summary':summary,'per_image':records}

        results = {'property':args.property,'teacher':teacher_id,'teacher_revision':revision,'split_counts':{s:len(v) for s,v in splits.items()},'runs':[],
                   'conditioning':'null class, decoder angle 0','steps':args.steps,'seeds':[673,674,675]}
        for split in ['test','unseen']:
            results[f'zero_{split}']=evaluate(None,splits[split])
            results[f'template_{split}']=evaluate(None,splits[split],fixed=template)
            results[f'target_reconstruction_{split}']=evaluate(None,splits[split],target_latent=True)
            if labels is not None: results[f'majority_{split}']=evaluate(None,splits[split],fixed=majority)
        if args.property == 'depth':
            for method in ['geodesic','addition']:
                for seed in [673,674,675]:
                    saved = torch.load(args.previous_results/f'{method}_seed{seed}.pt',map_location='cpu',weights_only=True)
                    edit = SharedTransform(saved['state_dict']['direction'].cuda(),method,model.spherify_mode).cuda()
                    edit.load_state_dict(saved['state_dict'])
                    results['runs'].append({'method':method,'seed':seed,'unseen':evaluate(edit,splits['unseen'],tag=f'{method}_unseen' if seed==673 else None)})
        else:
            initial = (ztargets[splits['train']]-zs[splits['train']]).mean(0,keepdim=True).cuda()
            for method in ['geodesic','addition']:
                for seed in [673,674,675]:
                    torch.manual_seed(seed)
                    edit = SharedTransform(initial+0.05*torch.randn_like(initial),method,model.spherify_mode).cuda()
                    opt = torch.optim.Adam([{'params':[edit.direction],'lr':0.03},{'params':[edit.raw_size],'lr':0.01}])
                    best,state,best_step = float('inf'),None,0
                    curve=[]; rng=torch.Generator().manual_seed(seed)
                    for step in range(args.steps+1):
                        if step%50==0 or step==args.steps:
                            val=evaluate(edit,splits['validation'])['summary']['rgb_mse']
                            curve.append({'step':step,'mse':val})
                            if val<best: best,state,best_step=val,copy.deepcopy(edit.state_dict()),step
                            print(f'{args.property} {method} seed={seed} step={step} val={val:.6f}',flush=True)
                        if step==args.steps: break
                        indices=torch.tensor(splits['train'])[torch.randint(len(splits['train']),(4,),generator=rng)]
                        opt.zero_grad(set_to_none=True)
                        loss=(decode_with_grad(model,edit(zs[indices].cuda()))-targets[indices].cuda()).square().mean()
                        if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
                        loss.backward()
                        if any(x.grad is None or not torch.isfinite(x.grad).all() for x in edit.parameters()): raise RuntimeError('Invalid edit gradients')
                        torch.nn.utils.clip_grad_norm_(edit.parameters(),1.)
                        opt.step()
                    edit.load_state_dict(state)
                    torch.save({'state_dict':state,'method':method,'seed':seed,'best_step':best_step},args.output_dir/f'{method}_seed{seed}.pt')
                    random_edit=SharedTransform(torch.randn_like(initial),method,model.spherify_mode).cuda()
                    random_edit.raw_size.data.copy_(edit.raw_size.data)
                    run={'method':method,'seed':seed,'curve':curve,'best_step':best_step,'size':edit.size().item(),
                         'train':evaluate(edit,splits['train']),
                         'test':evaluate(edit,splits['test'],tag=f'{method}_test' if seed==673 else None),
                         'unseen':evaluate(edit,splits['unseen'],tag=f'{method}_unseen' if seed==673 else None),
                         'random_test':evaluate(random_edit,splits['test']), 'random_unseen':evaluate(random_edit,splits['unseen'])}
                    results['runs'].append(run)
                    (args.output_dir/'results.json').write_text(json.dumps(results,indent=2))
            references=[]
            for index in splits['unseen'][::6][:4]:
                edit=SharedTransform((ztargets[index:index+1]-zs[index:index+1]).cuda(),'geodesic',model.spherify_mode).cuda()
                opt=torch.optim.Adam([{'params':[edit.direction],'lr':0.03},{'params':[edit.raw_size],'lr':0.01}])
                for step in range(150):
                    opt.zero_grad(set_to_none=True)
                    loss=(decode_with_grad(model,edit(zs[index:index+1].cuda()))-targets[index:index+1].cuda()).square().mean()
                    loss.backward(); torch.nn.utils.clip_grad_norm_(edit.parameters(),1.); opt.step()
                references.append(evaluate(edit,[index],tag=f'per_image_{index}'))
            results['per_image_references']=references
        results['frozen_verified']=not any(x.requires_grad or x.grad is not None for x in model.parameters())
        if not results['frozen_verified']: raise RuntimeError('Model was not frozen')
        (args.output_dir/'results.json').write_text(json.dumps(results,indent=2))
        print(f'{args.property} experiment completed',flush=True)
    finally:
        dist.destroy_process_group()


if __name__=='__main__': main()
