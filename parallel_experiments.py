"""Queue independent network/seed runs on separate GPUs (not DDP).

Run: python -u parallel_experiments.py --gpus 0,1,2,3,4,5,6,7 --run-dir PATH
Read progress: python parallel_experiments.py --status PATH
Stop queue and its children: python parallel_experiments.py --stop PATH
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import shutil

NETWORKS = ['CPDB', 'IREF', 'IREF_2015', 'MULTINET', 'PCNET', 'STRINGdb']


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def now():
    return datetime.now(timezone.utc).isoformat()


def build_jobs(indices, seeds):
    return [dict(id=f'{NETWORKS[i]}-seed{s}', network=NETWORKS[i], index=i,
                 seed=s, state='queued') for s in seeds for i in indices]


def read_metrics(run, job):
    path = run / 'jobs' / job['id'] / job['network'] / f"metrics{job['seed']}.json"
    return json.loads(path.read_text()) if path.exists() else None


def summarize(jobs):
    result = {}
    for network in NETWORKS:
        relevant = [j for j in jobs if j['network'] == network]
        successful = [j for j in relevant if j['state'] == 'succeeded']
        row = dict(completed=len(successful), requested=len(relevant),
                   complete=bool(relevant) and len(successful) == len(relevant))
        for metric in ('auprc', 'auroc'):
            values = [j['metrics'][f'test_{metric}_at_best_validation'] for j in successful]
            if values:
                row[metric] = dict(mean=statistics.mean(values), std=statistics.pstdev(values))
        result[network] = row
    return result


def free_memory(gpu):
    output = subprocess.check_output([
        'nvidia-smi', f'--id={gpu}', '--query-gpu=memory.free',
        '--format=csv,noheader,nounits'], text=True)
    return int(output.strip())


def snapshot(root, run):
    code = run / 'code'
    code.mkdir()
    hashes = {}
    for path in list(root.glob('*.py')) + [root / 'requirements.txt', root / 'KEGG_pathway_gene.txt']:
        shutil.copy2(path, code / path.name)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (code / 'data').symlink_to(root / 'data', target_is_directory=True)
    write_json(run / 'source_sha256.json', hashes)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--indices', default='0,1,2,3,4,5')
    parser.add_argument('--seeds', default='0,1,2,3,4')
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--status', type=Path)
    parser.add_argument('--stop', type=Path)
    args = parser.parse_args()
    if args.stop:
        if not (args.stop / 'manifest.json').is_file():
            parser.error('No experiment manifest at this path')
        (args.stop / 'STOP').touch()
        print('Stop requested. Scheduler will terminate its own workers.')
        return
    if args.status:
        manifest = json.loads((args.status / 'manifest.json').read_text())
        print('STATE:', manifest['state'], '| Updated:', manifest['updated_at'])
        for job in manifest['jobs']:
            m = job.get('metrics', {})
            print(job['id'], job['state'], 'GPU', job.get('gpu', '-'),
                  f"epoch {m.get('epochs_completed', 0)}/{manifest['epochs']}",
                  'sec/epoch', round(m.get('epoch_seconds', 0), 2))
        return
    if args.run_dir is None or args.epochs < 1:
        parser.error('--run-dir and positive --epochs are required')
    gpus = [int(x) for x in args.gpus.split(',')]
    indices = [int(x) for x in args.indices.split(',')]
    seeds = [int(x) for x in args.seeds.split(',')]
    if (len(set(gpus)) != len(gpus) or len(set(indices)) != len(indices)
            or len(set(seeds)) != len(seeds) or any(i not in range(6) for i in indices)
            or any(g < 0 for g in gpus)):
        parser.error('GPU IDs, network indices, and seeds must be unique and valid')
    root = Path(__file__).resolve().parent
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    if (run / 'manifest.json').exists() or (run / 'code').exists():
        parser.error('Run directory already used; choose a new directory')
    code = snapshot(root, run)
    jobs = build_jobs(indices, seeds)
    manifest = dict(state='running', created_at=now(), updated_at=now(),
                    pid=os.getpid(), epochs=args.epochs, gpus=gpus, jobs=jobs,
                    method='independent single-GPU experiments',
                    scope='DGHNN main model; not baseline or ablation experiments',
                    python=sys.version, python_hash_seed=0)
    (run / 'scheduler.pid').write_text(str(os.getpid()) + '\n')
    (run / 'environment.txt').write_text(subprocess.check_output(
        [sys.executable, '-m', 'pip', 'freeze'], text=True), encoding='utf-8')
    active = {}
    stop = False

    def request_stop(signum, frame):
        nonlocal stop
        stop = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, request_stop)

    def persist():
        manifest['updated_at'] = now()
        write_json(run / 'manifest.json', manifest)
        write_json(run / 'summary.json', summarize(jobs))

    try:
        persist()
        while any(j['state'] in ('queued', 'running') for j in jobs):
            if stop or (run / 'STOP').exists():
                manifest['state'] = 'stopped'
                break
            for gpu, (process, job, log) in list(active.items()):
                metrics = read_metrics(run, job)
                if metrics:
                    job['metrics'] = metrics
                code_return = process.poll()
                if code_return is not None:
                    job.update(returncode=code_return, ended_at=now())
                    valid = (metrics and metrics.get('complete')
                             and metrics.get('epochs_completed') == args.epochs)
                    job['state'] = 'succeeded' if code_return == 0 and valid else 'failed'
                    log.close()
                    del active[gpu]
                    print(now(), 'FINISHED', job['id'], job['state'], flush=True)
            for gpu in gpus:
                if gpu in active:
                    continue
                job = next((j for j in jobs if j['state'] == 'queued'), None)
                if job is None:
                    break
                if free_memory(gpu) < 20480:
                    continue
                jobdir = run / 'jobs' / job['id']
                jobdir.mkdir(parents=True)
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=str(gpu), DGHNN_PROFILE='server',
                           DGHNN_EDGE_CHUNK_SIZE='1048576', OMP_NUM_THREADS='4',
                           MKL_NUM_THREADS='4', PYTHONHASHSEED='0',
                           DGHNN_NUM_EPOCH=str(args.epochs), DGHNN_PATH_INDICES=str(job['index']),
                           DGHNN_SEEDS=str(job['seed']), DGHNN_OUTPUT_DIR=str(jobdir))
                log = (jobdir / 'train.log').open('w')
                process = subprocess.Popen([sys.executable, '-u', str(code / 'train.py')],
                    cwd=code, env=env, stdout=log, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL)
                job.update(state='running', pid=process.pid, gpu=gpu, started_at=now())
                active[gpu] = (process, job, log)
                print(now(), 'START', job['id'], 'GPU', gpu, 'PID', process.pid, flush=True)
            persist()
            time.sleep(5)
        if manifest['state'] == 'running':
            manifest['state'] = 'complete' if all(j['state'] == 'succeeded' for j in jobs) else 'failed'
    except BaseException as exc:
        manifest.update(state='failed', error=repr(exc))
        raise
    finally:
        for process, job, log in active.values():
            if process.poll() is None:
                process.terminate()
        for process, job, log in active.values():
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            job.update(state='stopped', ended_at=now())
            log.close()
        if manifest['state'] != 'complete':
            for job in jobs:
                if job['state'] == 'queued':
                    job['state'] = 'not_started'
        persist()
        print('SCHEDULER', manifest['state'], flush=True)


if __name__ == '__main__':
    main()
