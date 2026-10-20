import json
from pathlib import Path
import tempfile
import unittest

import torch

from experiment_io import save_checkpoint, write_json
from parallel_experiments import build_jobs, summarize


class ExperimentTests(unittest.TestCase):
    def test_queue_covers_six_networks_and_five_seeds_once(self):
        jobs = build_jobs(range(6), range(5))
        self.assertEqual(len(jobs), 30)
        self.assertEqual(len({j['id'] for j in jobs}), 30)
        self.assertEqual({(j['index'], j['seed']) for j in jobs},
                         {(i, s) for i in range(6) for s in range(5)})

    def test_summary_does_not_call_partial_results_complete(self):
        jobs = build_jobs([0], [0, 1])
        jobs[0].update(state='succeeded', metrics={
            'test_auprc_at_best_validation': .7, 'test_auroc_at_best_validation': .8})
        self.assertFalse(summarize(jobs)['CPDB']['complete'])
        jobs[1].update(state='succeeded', metrics={
            'test_auprc_at_best_validation': .9, 'test_auroc_at_best_validation': 1.})
        summary = summarize(jobs)['CPDB']
        self.assertTrue(summary['complete'])
        self.assertAlmostEqual(summary['auprc']['mean'], .8)
        self.assertAlmostEqual(summary['auprc']['std'], .1)

    def test_best_checkpoint_is_independent_and_contains_classifier(self):
        model, classifier = torch.nn.Linear(2, 2), torch.nn.Linear(2, 1)
        expected = model.weight.detach().clone()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'best.pth'
            state = save_checkpoint(path, model, classifier, epoch=7)
            with torch.no_grad():
                model.weight.add_(100)
                classifier.weight.add_(100)
            restored = torch.load(path, weights_only=True)
            torch.testing.assert_close(restored['model_state']['weight'], expected)
            torch.testing.assert_close(state['model_state']['weight'], expected)
            self.assertFalse(torch.equal(restored['classifier_state']['weight'], classifier.weight))
            self.assertEqual(restored['epoch'], 7)
            self.assertFalse(path.with_suffix('.pth.tmp').exists())

    def test_json_progress_is_readable_after_updates(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'progress.json'
            write_json(path, {'complete': False, 'epoch': 0})
            write_json(path, {'complete': True, 'epoch': 2999})
            self.assertEqual(json.loads(path.read_text()), {'complete': True, 'epoch': 2999})
            self.assertFalse(path.with_suffix('.json.tmp').exists())


if __name__ == '__main__':
    unittest.main()
