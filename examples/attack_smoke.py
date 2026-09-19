"""Tiny execution check of the real attack pipeline, not a paper experiment."""
import argparse
import json
import runpy
import sys
import tempfile
from pathlib import Path
import torch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    args=parser.parse_args()
    device=('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable')
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    root=Path(__file__).resolve().parents[1]
    (root/'runs').mkdir(exist_ok=True)
    output=Path(tempfile.mkdtemp(prefix='attack_smoke_',dir=root/'runs'))
    source=output/'synthetic_source'
    folder=source/'results/rank8_interval1'
    folder.mkdir(parents=True)
    prompt={'U':torch.randint(0,256,(77,8),dtype=torch.uint8),
            'V':torch.randint(0,256,(8,1024),dtype=torch.uint8),
            'U_scale':torch.tensor(1.0),'V_scale':torch.tensor(1.0),
            'U_zero_point':torch.tensor(0.0),'V_zero_point':torch.tensor(0.0)}
    torch.save(prompt,folder/'frame_00000.prompt')
    script=root/'research/attack_gated_tcn.py'
    sys.path.insert(0,str(script.parent))
    old_argv=sys.argv
    sys.argv=[str(script),'--source_frame_path',str(source),
              '--output_root',str(output/'attack'), '--data_root',str(root/'data/measured'),
              '--leak_counts','0,2','--channels','8','--num_blocks','2',
              '--max_dilation_power','1','--epochs','1','--min_optimizer_steps','1',
              '--batch_size','2','--attack_device',device,'--save_device','cpu',
              '--log_batch_every','0']
    try:
        runpy.run_path(str(script),run_name='__main__')
    finally:
        sys.argv=old_argv
    files=list((output/'attack/prompt_outputs').rglob('*.prompt'))
    if len(files)!=6:
        raise RuntimeError(f'Expected six condition outputs, found {len(files)}')
    for path in files:
        obj=torch.load(path,map_location='cpu',weights_only=True)
        assert set(obj)==set(prompt)
        assert obj['U'].shape==(77,8) and obj['V'].shape==(8,1024)
        for key in ['U_scale','V_scale','U_zero_point','V_zero_point']:
            assert torch.equal(obj[key],prompt[key])
    print(json.dumps({'device':device,'output_prompts':len(files),
                      'output':str(output.relative_to(root)),
                      'note':'Synthetic prompt, N=0/2, two blocks, one training step per model; no image inference.'}))


if __name__=='__main__':
    main()

