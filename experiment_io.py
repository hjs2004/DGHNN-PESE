"""Atomic experiment records and independent best-model snapshots."""
import json
import os
from pathlib import Path

import torch


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def save_checkpoint(path, model, classifier, **metadata):
    # CPU clones cannot change when subsequent optimizer steps update parameters.
    state = dict(metadata,
                 model_state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                 classifier_state={k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()})
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)
    return state
