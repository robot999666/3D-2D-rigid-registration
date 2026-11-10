"""项目流程命令入口。"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
FRAMES = ['02', '03', '04', '06', '07', '08', '09', '10']
SCRIPTS = {
    'prepare': '01_prepare_frame_data_and_models.py',
    'pose': '03_save_initial_pose_tview.py',
    'annotate': '04_annotate_full_liver_outer_mask.py',
    'samples': '05_render_training_pose_pairs.py',
    'sample-depth': '06_generate_depth_for_training_samples.py',
    'depth': '07_generate_depth_for_real_frame.py',
    'train': '08_train_refinenet_pose_model.py',
    'infer': '09_infer_pose_and_export_stl.py',
}

def profile_env():
    # 统一由配置文件设置项目路径和图像参数。
    env = {k: v for k, v in os.environ.items() if not k.startswith(('AR_', 'DA2_')) and k not in {
        'PATIENT_ID', 'FRAME_ID', 'DATA_ROOT', 'CASE_ROOT', 'RESULT_ROOT', 'WEIGHTS_ROOT'}}
    for line in (ROOT / 'configs/patient1_fullres.env').read_text('utf-8').splitlines():
        if line.strip() and not line.lstrip().startswith('#'):
            key, value = line.split('=', 1)
            env[key.strip()] = value.strip()
    env.update({
        'AR_PROJECT_ROOT': str(ROOT), 'DATA_ROOT': str(ROOT / 'data/raw'),
        'CASE_ROOT': str(ROOT / 'data/processed'), 'RESULT_ROOT': str(ROOT / 'results/run'),
        'WEIGHTS_ROOT': str(ROOT / 'weights'), 'DA2_PROJECT_DIR': str(ROOT / 'modules'),
        'DA2_CHECKPOINT_PATH': str(ROOT / 'weights/depth_anything_v2_vitl.pth'),
        'AR_WEIGHT_PATH': str(ROOT / 'weights/refinenet_patient1_fullres_best.pth'),
        'PYTHONIOENCODING': 'utf-8', 'PYTHONDONTWRITEBYTECODE': '1',
    })
    if os.name == 'nt':
        cache = Path('D:/environment/cache')
        for key, folder in [('MPLCONFIGDIR','matplotlib'), ('TORCH_HOME','torch'), ('TEMP','tmp'), ('TMP','tmp')]:
            target = cache / folder
            target.mkdir(parents=True, exist_ok=True)
            env[key] = str(target)
    return env

def invoke(script, env, *args):
    subprocess.run([sys.executable, str(ROOT / 'scripts' / script), *map(str,args)],
                   env=env, cwd=ROOT, check=True)

def check(deep=False):
    errors = []
    report = {'python': sys.version.split()[0], 'executable': sys.executable, 'dependencies': {}}
    for name in ('numpy', 'scipy', 'matplotlib', 'cv2', 'open3d', 'torch', 'torchvision', 'tqdm'):
        try:
            module = importlib.import_module(name)
            report['dependencies'][name] = module.__version__
        except ImportError as exc:
            errors.append(f'{name}: {exc}')
    if 'torch' in sys.modules:
        import torch
        report['cuda_available'] = torch.cuda.is_available()
        report['gpu'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    required = ['weights/depth_anything_v2_vitl.pth', 'weights/refinenet_patient1_fullres_best.pth']
    for frame in FRAMES:
        required += [f'data/processed/Patient1/{frame}/T_view.txt',
                     f'data/processed/Patient1/{frame}/masks/full_liver_mask.png',
                     f'data/processed/Patient1/{frame}/masks/depth_valid_mask.png',
                     f'data/raw/Dataset/Patient1/Lap Images/{frame}.png',
                     f'data/raw/Dataset/Patient1/LUS Calibration and Pose/{frame}.mat',
                     f'data/raw/Dataset/Patient1/LUS Segmentation/json/{frame}.json']
    errors += [f'Missing: {p}' for p in required if not (ROOT / p).is_file()]
    if not (ROOT / 'weights/depth_anything_v2_vitl.pth').is_file():
        sys.path.insert(0, str(ROOT / 'modules'))
        from shared.da2_engine import missing_checkpoint_message
        errors.append(missing_checkpoint_message('vitl', ROOT / 'weights/depth_anything_v2_vitl.pth'))
    if deep and not errors:
        sys.path.insert(0, str(ROOT / 'modules'))
        from shared.intraop_masks import validate_case_masks
        from learning.models.refine_network import RefineNet
        from shared.da2_engine import DepthAnythingEstimator
        class Cfg:
            use_BN = True
            rot_rep = '6d'
        for frame in FRAMES:
            validate_case_masks(str(ROOT / 'data/processed/Patient1' / frame), (1080,1920))
        state = torch.load(ROOT / 'weights/refinenet_patient1_fullres_best.pth', map_location='cpu', weights_only=True)
        model = RefineNet(cfg=Cfg(), c_in=5)
        model.load_state_dict(state, strict=True)
        report['refinenet_strict_load'] = True
        report['feature_token_grid'] = model.feature_token_grid
        estimator = DepthAnythingEstimator(device='cpu')
        report['depth_model_strict_load'] = True
        del estimator, model, state
    report['errors'] = errors
    report['ok'] = not errors
    out = ROOT / 'results/verification'
    out.mkdir(parents=True, exist_ok=True)
    filename = 'environment_check_deep.json' if deep else 'environment_check.json'
    (out / filename).write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if errors:
        raise SystemExit(1)

def evaluate(env, debug=False, frames='all'):
    patient = 'Patient1'
    source = ROOT / 'experiments/patient1_fullres/result'
    out = ROOT / 'results/evaluation' / patient
    if frames!='all':
        out = out / ('frames_' + frames.replace(',', '_'))
    extra = ['--include-debug'] if debug else []
    invoke('11_evaluate_registration.py', env, '--patient-id',patient,
           '--data-root', ROOT / 'data/raw', '--result-root',source,'--frames',frames,'--output-dir',out,*extra)
    if frames=='all':
        result = json.loads((out / 'Patient1_final_global_evaluation.json').read_text('utf-8'))
        assert result['frame_count']==8 and abs(result['TRE_mm']-8.9490400807)<1e-6 and result['IC']==0, result
        print('Archived Patient1 global metric matches the saved result.')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('check', help='Check runtime assets and dependencies').add_argument('--deep',action='store_true')
    for name in ('evaluate',):
        sub = commands.add_parser(name, help='Evaluate archived registration outputs')
        sub.add_argument('--frames', default='all', help='Comma-separated frame numbers or all')
        sub.add_argument('--include-debug', action='store_true')
    for name in ('infer','prepare','pose','annotate','depth'):
        sub = commands.add_parser(name)
        sub.add_argument('--frames', default='02', help='Comma-separated Patient1 frames or all')
        if name=='infer':
            sub.add_argument('--weight', type=Path, help='RefineNet checkpoint; defaults to the delivered model')
    for name in ('samples','sample-depth','train'):
        sub = commands.add_parser(name)
        if name=='train':
            sub.add_argument('--batch-size',type=int)
    args = parser.parse_args()
    env = profile_env()
    os.environ.clear()
    os.environ.update(env)
    if args.command=='check':
        check(args.deep)
        return
    if args.command.startswith('evaluate'):
        allowed = FRAMES
        requested = args.frames.split(',')
        if args.frames!='all' and any(f not in allowed for f in requested):
            parser.error('Available frames: '+','.join(allowed))
        evaluate(env,debug=args.include_debug,frames=args.frames)
        return
    frame_text = getattr(args, 'frames', '02')
    frames = FRAMES if frame_text=='all' else frame_text.split(',')
    if not frames or any(f not in FRAMES for f in frames):
        parser.error('Available Patient1 frames: '+','.join(FRAMES))
    if args.command in ('samples','sample-depth','train') and frames!=['02']:
        parser.error('The included training profile uses reference frame 02.')
    if getattr(args, 'batch_size', None) is not None:
        if args.batch_size<1:
            parser.error('--batch-size must be positive')
        env['AR_BATCH_SIZE']=str(args.batch_size)
    if args.command=='pose':
        env['AR_TVIEW_CONFIG']=str(ROOT / 'data/processed/Patient1/02/initial_pose.json')
    if args.command=='infer' and args.weight is not None:
        weight = args.weight.resolve()
        if not weight.is_file():
            parser.error('Checkpoint does not exist: '+str(weight))
        env['AR_WEIGHT_PATH']=str(weight)
    if args.command in ('depth', 'sample-depth', 'infer'):
        sys.path.insert(0, str(ROOT / 'modules'))
        from shared.da2_engine import resolve_checkpoint
        try:
            resolve_checkpoint()
        except FileNotFoundError as exc:
            parser.exit(2, str(exc)+'\n')
    for frame in frames:
        env['FRAME_ID']=frame
        if args.command=='infer':
            invoke(SCRIPTS['depth'],env)
        invoke(SCRIPTS[args.command],env)

if __name__=='__main__':
    main()
