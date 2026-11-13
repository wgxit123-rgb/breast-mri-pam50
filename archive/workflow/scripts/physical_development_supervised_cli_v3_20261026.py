"""Small CLI correction over preserved v2: propagate failed child exit status.

No model/data work at import. Any future patient execution must bind this entry
along with v2 and the original runner in a new adopted execution protocol.
"""
import argparse
import io
import json
from pathlib import Path
import sys
import unittest

import physical_development_supervised_v2_20261026 as adapter


def exit_status(result):
    if result.get('timed_out'):
        return 124
    if result.get('status') != 'completed' or result.get('exit_code') != 0:
        code = result.get('exit_code')
        return code if isinstance(code, int) and 1 <= code <= 255 else 1
    if result.get('owned_active_processes_after_termination') != 0 or result.get('job_closed') is not True:
        return 1
    return 0


def synthetic_handshake():
    raw = sys.stdin.readline(32769)
    if not raw or len(raw) > 32768 or json.loads(raw).get('scope') != 'synthetic_v3_exit_status_only':
        raise ValueError('Synthetic supervisor handshake required')


def selftest(output):
    synthetic_handshake()
    output = adapter.inside(adapter.ROOT, output)
    if output.exists():
        raise FileExistsError('No synthetic correction-test retry')
    artifacts = output.parent/'v3_exit_fixtures'
    artifacts.mkdir(exist_ok=False)

    class ExitStatusTests(unittest.TestCase):
        def test_clean_completion(self):
            self.assertEqual(exit_status(dict(status='completed',exit_code=0,timed_out=False,
                owned_active_processes_after_termination=0,job_closed=True)),0)

        def test_error_and_timeout_not_success(self):
            self.assertEqual(exit_status(dict(status='worker_failed',exit_code=7)),7)
            self.assertEqual(exit_status(dict(status='hard_timeout',exit_code=124,timed_out=True)),124)
            self.assertEqual(exit_status(dict(status='startup_failed')),1)

        def test_cleanup_not_reported_complete_if_missing(self):
            self.assertEqual(exit_status(dict(status='completed',exit_code=0)),1)
            self.assertEqual(exit_status(dict(status='completed',exit_code=0,
                owned_active_processes_after_termination=1,job_closed=True)),1)

        def test_real_failed_worker_code_propagated(self):
            result = adapter.owned([sys.executable,'-X','utf8','-B',str(Path(__file__).resolve()),'fixture-fail'],
                dict(scope='synthetic_v3_exit_status_only'),adapter.ROOT,artifacts/'failed_worker',5.)
            self.assertEqual(result['status'],'worker_failed')
            self.assertEqual(result['exit_code'],7)
            self.assertEqual(exit_status(result),7)
            self.assertEqual(result['owned_active_processes_after_termination'],0)

    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ExitStatusTests))
    with output.open('x',encoding='utf-8') as stream:
        json.dump(dict(tests_run=result.testsRun,successful=result.wasSuccessful(),
                       failures=len(result.failures),errors=len(result.errors),
                       scope='v3_exit_status_correction_only',patient_training=False,
                       old9tests_repeated=False,details=log.getvalue()),stream,indent=2)
    print(log.getvalue(),flush=True)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    # Normal check/train flags are intentionally the same as preserved v2.
    if len(sys.argv) > 1 and sys.argv[1] == 'fixture-fail':
        synthetic_handshake()
        print('EXPECTED_SYNTHETIC_CHILD_EXIT7',flush=True)
        raise SystemExit(7)
    if len(sys.argv) > 1 and sys.argv[1] == 'selftest':
        parser = argparse.ArgumentParser()
        parser.add_argument('mode')
        parser.add_argument('--output',required=True)
        selftest(parser.parse_args().output)
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        parser = argparse.ArgumentParser()
        parser.add_argument('mode')
        parser.add_argument('--plan',required=True)
        parser.add_argument('--plan-sha256',required=True)
        parser.add_argument('--output',required=True)
        parser.add_argument('--arm',required=True)
        parser.add_argument('--fold',type=int,required=True)
        parser.add_argument('--gpu',type=int,required=True)
        args = parser.parse_args()
        result = adapter.launch(adapter.ROOT,args.plan,args.plan_sha256,args.arm,args.fold,args.gpu,
                                adapter.inside(adapter.ROOT,args.output))
        print(json.dumps(result),flush=True)
        raise SystemExit(exit_status(result))
    adapter.main()
