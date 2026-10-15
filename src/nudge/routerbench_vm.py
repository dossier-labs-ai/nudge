"""Dedicated benchmark VM supervisor: provider lifecycle, exports and teardown.

Management requests are not inference. All inference stays behind TensorZero.
Never supplies credentials in command arguments, files, or environment variables.
"""

import argparse
import json
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

from .routerbench_exports import atomic_text
from .routerbench_live import secret


class Baseten:
    def __init__(self, model, environment, key, base='https://api.baseten.co/v1'):
        self.model, self.environment, self.key, self.base = model, environment, key, base
        self.deployment = None

    def request(self, method, suffix, body=None):
        url = f'{self.base}/models/{self.model}/{suffix}'
        request = urllib.request.Request(url, method=method,
            data=json.dumps(body).encode() if body is not None else None,
            headers={'Authorization': f'Api-Key {self.key}', 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f'Baseten management {method} {suffix}: HTTP {error.code}') from None

    def inspect(self):
        value = self.request('GET', f'environments/{self.environment}')
        deployment = value['current_deployment']['id']
        if self.deployment is not None and deployment != self.deployment:
            raise RuntimeError('environment deployment changed during run')
        self.deployment = deployment
        return value

    def deployment_path(self):
        if self.deployment is None:
            self.inspect()
        return f'deployments/{self.deployment}'

    def action(self, action):
        if action not in ('activate', 'deactivate'):
            raise ValueError('unsupported environment action')
        return self.request('POST', f'{self.deployment_path()}/{action}', {})

    def set_minimum(self, minimum):
        return self.request('PATCH', f'{self.deployment_path()}/autoscaling_settings',
                            {'min_replica': minimum})


def publish(root, uri, *, full=False):
    root = Path(root)
    if not uri.startswith('gs://'):
        raise ValueError('output must be a GCS URI')
    files = [p for p in (root / 'exports').glob('*') if p.suffix in ('.json', '.csv', '.png')]
    if files:
        subprocess.run(['gcloud', 'storage', 'cp', *map(str, files), uri.rstrip('/') + '/live/'],
                       check=True, timeout=180)
    if full:
        # Keep output outside the archived tree: creating it inside changes the
        # directory during tar and can abort upload with "file changed as we read it".
        archive = root.with_name(root.name + '-artifacts.tar.gz')
        subprocess.run(['tar', '-czf', str(archive), '--exclude=artifacts.tar.gz',
                        '-C', str(root), '.'], check=True, timeout=300)
        subprocess.run(['gcloud', 'storage', 'cp', str(archive), uri.rstrip('/') + '/artifacts.tar.gz'],
                       check=True, timeout=600)


def stop_process(process):
    if process is not None and process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def supervise(args, provider):
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / 'exports').mkdir(exist_ok=True)
    worker = bridge = None
    gateway_started = False
    lifecycle = {'model': args.model, 'environment': args.environment,
                 'output_uri': args.output_uri, 'stage': 'starting', 'resume': args.resume}
    exit_code = 1
    endpoint = f'https://model-{args.model}.api.baseten.co/environments/{args.environment}/sync/v1/systemone'
    config = args.gateway_config.resolve()
    def heartbeat():
        lifecycle['heartbeat_at'] = datetime.now(timezone.utc).isoformat()
        atomic_text(root / 'exports' / 'lifecycle.json', json.dumps(lifecycle, indent=2))

    try:
        if args.resume and not (root / 'run' / 'manifest.json').exists():
            raise RuntimeError('resume requires an existing run manifest')
        with (root / 'executions.jsonl').open('a') as log:
            log.write(json.dumps({**lifecycle, 'started_at': datetime.now(timezone.utc).isoformat(),
                                  'hydration': 'bounded-concurrent-v1'}) + '\n')
        before = provider.inspect()
        atomic_text(root / 'baseten-before.json', json.dumps(before, indent=2))
        lifecycle['stage'] = 'activating'
        heartbeat()
        if before.get('current_deployment', {}).get('status') in ('INACTIVE', 'DEACTIVATED'):
            provider.action('activate')
        provider.set_minimum(1)
        # Readiness is checked through real TensorZero probe, with a bounded wait.
        with (root / 'bridge.log').open('a') as log:
            bridge = subprocess.Popen([sys.executable, '-m', 'nudge.rune_gateway',
                '--endpoint=' + endpoint, '--max-concurrency=16'], stdout=log, stderr=log)
        if args.resume:
            # Exact task-owned container only; no force, so an active container
            # fails closed rather than interrupting another supervisor.
            subprocess.run(['docker', 'rm', 'rune-bench-gateway'], check=True, timeout=60)
        subprocess.run(['docker', 'run', '-d', '--name=rune-bench-gateway',
            '--network=host', '--add-host=host.docker.internal:127.0.0.1',
            '-v', str(config.parent) + ':/app/config:ro', args.gateway_image,
            '--config-file=/app/config/routerbench.toml', '--bind-address=127.0.0.1:3000'],
            check=True, timeout=180)
        gateway_started = True
        common = [sys.executable, '-m', 'nudge.routerbench',
            '--prepared=' + str(args.prepared.resolve()), '--seed=20261014',
            '--cache-dir=' + str(root / 'cache'), '--gateway-config=' + str(config),
            '--gateway-url=http://127.0.0.1:3000', '--rune-endpoint=' + endpoint,
            '--rune-revision=' + before['current_deployment']['id'],
            '--rune-secret-project=' + args.secret_project, '--rune-secret=' + args.secret,
            '--glm-secret-project=' + args.secret_project, '--glm-secret=' + args.secret,
            '--inference-concurrency=16', '--max-training-rows=1024']
        deadline = time.monotonic() + 1200
        lifecycle['stage'] = 'probing'
        while True:
            heartbeat()
            publish(root, args.output_uri)
            with (root / 'probe.log').open('a') as log:
                probe = subprocess.run([*common, 'probe', '--run-dir=' + str(root / 'probe')],
                                       stdout=log, stderr=log, timeout=600)
            if probe.returncode == 0:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError('provider readiness probe failed within startup allowance')
            time.sleep(15)
        lifecycle['stage'] = 'running'
        with (root / 'run.log').open('a') as log:
            worker = subprocess.Popen([*common, 'run', '--run-dir=' + str(root / 'run'),
                '--export-dir=' + str(root / 'exports'), *(['--resume'] if args.resume else [])],
                stdout=log, stderr=log)
        deadline = time.monotonic() + args.max_hours * 3600
        while worker.poll() is None:
            heartbeat()
            publish(root, args.output_uri)
            if time.monotonic() >= deadline:
                raise TimeoutError('maximum run duration reached')
            time.sleep(30)
        exit_code = worker.returncode
        lifecycle['stage'] = 'complete' if exit_code == 0 else 'failed'
    except BaseException as error:
        lifecycle.update(stage='failed', error=type(error).__name__,
                         detail=str(error) if isinstance(error, RuntimeError) else None)
        print(json.dumps(lifecycle), flush=True)
    finally:
        for process in (worker, bridge):
            try:
                stop_process(process)
            except Exception as error:
                lifecycle['process_cleanup_error'] = type(error).__name__
        if gateway_started:
            try:
                subprocess.run(['docker', 'stop', 'rune-bench-gateway'], timeout=60, check=False)
            except Exception as error:
                lifecycle['gateway_cleanup_error'] = type(error).__name__
        # Retry cleanup independently of worker/export failures.
        for attempt in range(5):
            try:
                try:
                    provider.set_minimum(0)
                except Exception as error:
                    lifecycle['minimum_restore_error'] = type(error).__name__
                provider.action('deactivate')
                after = provider.inspect()
                atomic_text(root / 'baseten-after.json', json.dumps(after, indent=2))
                lifecycle['deactivation_requested'] = True
                lifecycle['provider_status_after_request'] = after.get('current_deployment', {}).get('status')
                break
            except Exception as error:
                lifecycle['cleanup_error'] = type(error).__name__
                time.sleep(10)
        lifecycle['exit_code'] = exit_code
        heartbeat()
        try:
            publish(root, args.output_uri, full=True)
        finally:
            if args.shutdown:
                subprocess.run(['shutdown', '-h', 'now'], check=True, timeout=30)
    return exit_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('inspect', 'run', 'sync'))
    parser.add_argument('--model', help='Baseten model id hosting Rune.')
    parser.add_argument('--environment', default='eval', help='Baseten environment name.')
    parser.add_argument('--secret-project', help='Secret Manager project holding the Baseten API key.')
    parser.add_argument('--secret', help='Secret Manager reference NAME[:VERSION] for the Baseten API key.')
    parser.add_argument('--credential-source', default='gcloud', choices=('gcloud', 'application-default'))
    parser.add_argument('--root', type=Path)
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--gateway-config', type=Path)
    parser.add_argument('--gateway-image')
    parser.add_argument('--output-uri')
    parser.add_argument('--max-hours', type=float, default=24)
    parser.add_argument('--shutdown', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.command == 'sync':
        if not args.root or not args.output_uri:
            parser.error('sync requires root and output-uri')
        args.root.mkdir(parents=True, exist_ok=True)
        while True:
            subprocess.run(['gcloud', 'storage', 'rsync', args.output_uri.rstrip('/') + '/live',
                            str(args.root)], check=True, timeout=180)
            state = args.root / 'lifecycle.json'
            if state.exists() and json.loads(state.read_text()).get('deactivation_requested'):
                break
            time.sleep(30)
        return
    if not all((args.model, args.secret_project, args.secret)):
        parser.error('inspect and run require model, secret-project and secret')
    provider = Baseten(args.model, args.environment,
                       secret(args.secret_project, args.secret, args.credential_source))
    if args.command == 'inspect':
        print(json.dumps(provider.inspect(), indent=2))
        return
    if not all((args.root, args.prepared, args.gateway_config, args.gateway_image,
                args.output_uri)) or not args.output_uri.startswith('gs://') or args.max_hours <= 0:
        parser.error('run requires root, prepared, gateway-config, gateway-image, GCS output-uri and positive max-hours')
    def interrupt(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    raise SystemExit(supervise(args, provider))


if __name__ == '__main__':
    main()
