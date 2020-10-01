# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""
import torch
from ray import tune

from fastreid.engine.hooks import EvalHook, flatten_results_dict


class TuneReportHook(EvalHook):
    def _do_eval(self):
        results = self._func()

        if results:
            assert isinstance(
                results, dict
            ), "Eval function must return a dict. Got {} instead.".format(results)

            flattened_results = flatten_results_dict(results)
            for k, v in flattened_results.items():
                try:
                    v = float(v)
                except Exception:
                    raise ValueError(
                        "[EvalHook] eval_function should return a nested dict of float. "
                        "Got '{}: {}' instead.".format(k, v)
                    )

        # Remove extra memory cache of main process due to evaluation
        torch.cuda.empty_cache()

        tune.report(r1=results['Rank-1'], map=results['mAP'], score=(results['Rank-1'] + results['mAP']) / 2)
