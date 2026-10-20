"""Short full-graph benchmarks; never overwrite training results.

Run on a free GPU with CUDA_VISIBLE_DEVICES=... python -u benchmark_server.py.
Uses the training script's data preparation and one split/seed, not toy data.
"""
import ast
import gc
import io
import json
import os
from pathlib import Path
import tempfile
import time
from contextlib import redirect_stdout

os.environ.setdefault("DGHNN_PATH_LIMIT", "1")
os.environ.setdefault("DGHNN_SEEDS", "0")


def benchmark(scope):
    import torch
    from model import ChunkedGCNConv, GCN, HyperGCN

    model, classifier = scope["model"], scope["classifier"]
    originals = [{k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                 for net in (model, classifier)]
    variants = [("baseline", 32768, True),
                ("chunk262k", 262144, True),
                ("chunk1m", 1048576, True),
                ("no_checkpoint", 262144, False)]
    selected = os.environ.get("DGHNN_BENCH_VARIANTS", "")
    if selected:
        variants = [v for v in variants if v[0] in selected.split(",")]
    report = []
    print("GRAPH", scope["features"].shape, scope["edge_index"].shape,
          "TRAIN_NODES", len(scope["idx_train"]), flush=True)
    for name, chunk, checkpoint in variants:
        print("BENCH_START", name, flush=True)
        optimizer = None
        try:
            for net, state in zip((model, classifier), originals):
                net.load_state_dict(state)
                net.zero_grad(set_to_none=True)
            for layer in model.modules():
                if isinstance(layer, ChunkedGCNConv):
                    layer.edge_chunk_size = chunk
                if isinstance(layer, (GCN, HyperGCN)):
                    layer.gradient_checkpointing = checkpoint
            scope["set_seed"](0)
            optimizer = torch.optim.Adam(
                list(model.parameters()) + list(classifier.parameters()),
                lr=scope["LEARNING_RATE"], weight_decay=1e-5)
            scaler = torch.amp.GradScaler("cuda")
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            rows = []
            for epoch in range(int(os.environ.get("DGHNN_BENCH_EPOCHS", "2"))):
                scope["result"] = io.StringIO()
                torch.cuda.synchronize()
                start = time.perf_counter()
                with redirect_stdout(scope["result"]):
                    scope["train"](model, optimizer, scope["criterion"], classifier,
                        scope["features"], scope["edge_index"], scope["h_edge_index"],
                        scope["meta_y"], scope["idx_train"], scaler)
                    val = scope["test"](model, scope["criterion"], classifier,
                        scope["features"], scope["edge_index"], scope["h_edge_index"],
                        scope["meta_y"], scope["idx_val"])
                    test = scope["test"](model, scope["criterion"], classifier,
                        scope["features"], scope["edge_index"], scope["h_edge_index"],
                        scope["meta_y"], scope["idx_test"])
                torch.cuda.synchronize()
                row = dict(epoch=epoch, seconds=time.perf_counter()-start,
                    allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                    validation=val, test=test, scaler_scale=scaler.get_scale())
                rows.append(row)
                print("BENCH_EPOCH", name, json.dumps(row), flush=True)
            report.append(dict(name=name, chunk=chunk, checkpoint=checkpoint,
                               status="ok", epochs=rows))
        except torch.cuda.OutOfMemoryError as exc:
            print("BENCH_OOM", name, str(exc), flush=True)
            report.append(dict(name=name, status="oom"))
        finally:
            model.zero_grad(set_to_none=True)
            classifier.zero_grad(set_to_none=True)
            del optimizer
            gc.collect()
            torch.cuda.empty_cache()
    print("BENCH_RESULT", json.dumps(report), flush=True)


if __name__ == "__main__":
    source = Path(__file__).with_name("train.py")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    # Keep data preparation and model construction exactly as in train.py.
    split_loop = next(n for n in tree.body if isinstance(n, ast.For)
                      and isinstance(n.target, ast.Name) and n.target.id == "path_index")
    seed_loop = next(n for n in split_loop.body if isinstance(n, ast.For)
                     and isinstance(n.target, ast.Name) and n.target.id == "seed")
    cut = next(i for i, n in enumerate(seed_loop.body) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "result" for t in n.targets))
    seed_loop.body[cut:] = ast.parse("benchmark(globals())\nraise SystemExit(0)").body
    # Redirect even the setup's output directory creation away from user runs.
    with tempfile.TemporaryDirectory(prefix="dghnn-bench-") as temp:
        assign = next(n for n in tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "save_dir" for t in n.targets))
        assign.value = ast.Constant(temp)
        ast.fix_missing_locations(tree)
        scope = dict(__name__="__benchmark__", __file__=str(source), benchmark=benchmark)
        print("Loading full project data for isolated benchmark...", flush=True)
        exec(compile(tree, str(source), "exec"), scope)
