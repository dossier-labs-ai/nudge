import argparse
import json
import subprocess
import tarfile
import tempfile
import threading
import unittest
from types import SimpleNamespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from nudge.routerbench_vm import Baseten, publish, supervise


class VMTests(unittest.TestCase):
    def test_final_archive_is_created_outside_run_tree_and_uploaded(self):
        run = subprocess.run
        uploads = []
        def external(args, **kwargs):
            if args[0] == 'gcloud':
                uploads.append(args)
                return SimpleNamespace(returncode=0)
            return run(args, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'run'
            (root / 'exports').mkdir(parents=True)
            (root / 'evidence.json').write_text('{"preserved":true}')
            with patch('subprocess.run', side_effect=external):
                publish(root, 'gs://fixture/run', full=True)
            archive = root.with_name('run-artifacts.tar.gz')
            with tarfile.open(archive) as saved:
                self.assertEqual(json.load(saved.extractfile('./evidence.json')), {'preserved': True})
            self.assertEqual(uploads[-1][-1], 'gs://fixture/run/artifacts.tar.gz')

    def test_provider_and_failed_startup_cleanup_use_external_boundaries(self):
        requests = []
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                requests.append(('GET', self.path, None))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"current_deployment":{"id":"test","status":"INACTIVE"}}')

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append((self.command, self.path, body))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"success":true}')

            do_PATCH = do_POST

            def log_message(self, *_):
                pass

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        commands = []
        def command(args, **kwargs):
            commands.append(args)
            return None
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args = argparse.Namespace(root=root, model='test', environment='eval',
                    output_uri='gs://test/run', gateway_config=root / 'gateway.toml',
                    gateway_image='test', secret_project='test', secret='key:1',
                    prepared=root / 'prepared.json', max_hours=1, shutdown=True, resume=False)
                provider = Baseten('test', 'eval', 'fixture-key',
                                   base=f'http://127.0.0.1:{server.server_port}')
                # Process launch is the external OS boundary. Failure must still
                # deactivate the real HTTP fixture and attempt artifact upload.
                with patch('subprocess.Popen', side_effect=OSError('cannot start')):
                    with patch('subprocess.run', side_effect=command):
                        self.assertEqual(supervise(args, provider), 1)
                lifecycle = json.loads((root / 'exports' / 'lifecycle.json').read_text())
                self.assertTrue(lifecycle['deactivation_requested'])
                self.assertEqual(lifecycle['stage'], 'failed')
                self.assertTrue(any('/activate' in path for _, path, _ in requests))
                self.assertTrue(any('/deactivate' in path for _, path, _ in requests))
                self.assertTrue(any(body == {'min_replica': 1} for _, _, body in requests))
                self.assertTrue(any(path == '/models/test/deployments/test/autoscaling_settings'
                                    for _, path, _ in requests))
                self.assertEqual(commands[-1], ['shutdown', '-h', 'now'])
                self.assertTrue(any(c[:3] == ['gcloud', 'storage', 'cp'] for c in commands))
                args.resume = True
                (root / 'run').mkdir()
                (root / 'run' / 'manifest.json').write_text('{}')
                commands.clear()
                def complete(args, **kwargs):
                    commands.append(args)
                    return SimpleNamespace(returncode=0, poll=lambda: 0)
                with patch('subprocess.Popen', side_effect=complete), patch('subprocess.run', side_effect=complete):
                    self.assertEqual(supervise(args, provider), 0)
                worker = next(c for c in commands if 'nudge.routerbench' in c and 'run' in c)
                self.assertIn('--resume', worker)
                self.assertIn('--run-dir=' + str(root.resolve() / 'run'), worker)
                remove = commands.index(['docker', 'rm', 'rune-bench-gateway'])
                launch = next(i for i, c in enumerate(commands) if c[:2] == ['docker', 'run'])
                self.assertLess(remove, launch)
                lifecycle = json.loads((root / 'exports' / 'lifecycle.json').read_text())
                self.assertEqual(lifecycle['stage'], 'complete')
                self.assertTrue(lifecycle['deactivation_requested'])
                self.assertTrue(json.loads((root / 'executions.jsonl').read_text().splitlines()[-1])['resume'])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
