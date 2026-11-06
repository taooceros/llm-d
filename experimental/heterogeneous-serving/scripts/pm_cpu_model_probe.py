#!/usr/bin/env python3
"""Local CPU-only verification of the native CPU-model boundary (Gemma4, TP8 geometry, no TPU).

Requires pm_cpu_model_sdk.patch applied to the CPU SDK, pm_rewind_cpu_model.py, and a private
non-writable copy of the worker CPU-prefix sources (the Ray launcher trust check refuses others).

  pm_cpu_model_probe.py export DIR          real API CLI config -> DIR/model-config.pkl
  pm_cpu_model_probe.py reference DIR RANK  ordinary CPU construction + real post-load, for comparison
  pm_rewind.py prepare CACHE --factory pm_cpu_model_probe:prepare ... ; pm_rewind.py run CACHE ... -- check
      native capture of rank 0 and 5 pure-meta holders, restore in a new PID, fail-closed checks,
      CPU materialization and real post-load of rank 5 (TPU materialization is not exercised here)

Environment: PM_REWIND_SERVE_ARGV (export), PM_CPU_MODEL_CONFIG (reference/native),
PM_CPU_MODEL_PROOF (native proof directory). The export imports Ray but never connects;
everything else must stay Ray-free with TPU, JAX and torch.distributed uninitialized.
"""
import json
import os
from pathlib import Path
import sys
import time
import traceback

import pm_ray_zygote as workers
import pm_rewind_cpu_model as cpu_model
import pm_rewind_ray as factory

RANKS = (0, 5)
TP = 8
LIFECYCLE_ROLES = ('live', 'empty_runtime', 'postload_slot', 'reload_metadata', 'stage_missing', 'unused_constant')
SCOPE = 'pure-meta rank holders and real Pallas registration; no weights, AOT, TPU executable or inference'


def runtime_state():
    torch = sys.modules.get('torch')
    loader = sys.modules.get('torch_tpu._loader')
    bridge = sys.modules.get('jax._src.xla_bridge')
    return {
        'ray_imported': 'ray' in sys.modules,
        'raylet_imported': 'ray._raylet' in sys.modules,
        'ray_initialized': bool(sys.modules.get('ray') and sys.modules['ray'].is_initialized()),
        'tpu_initialized': torch.tpu.is_initialized() if torch and hasattr(torch, 'tpu') else None,
        'tpu_bad_fork': torch.tpu._is_in_bad_fork() if torch and hasattr(torch, 'tpu') else None,
        'loader_flags': {k: loader.__dict__.get(k) for k in ('_LOADED', '_DEVICE_INITIALIZED')} if loader else None,
        'jax_backends': list(bridge.__dict__.get('_backends', {})) if bridge else [],
        'jax_default_backend': str(bridge.__dict__.get('_default_backend')) if bridge else None,
        'distributed_initialized': bool(torch and torch.distributed.is_initialized()),
    }


def checked(stages, name, fn, *, ray_free=True):
    started = time.perf_counter()
    result = fn()
    workers.assert_cpu_boundary(ray_free=ray_free, loader_unloaded=True)
    stages.append({'name': name, 'seconds': time.perf_counter() - started, 'after': runtime_state()})
    return result


def write_report(path, report):
    with path.open('x') as f:
        json.dump(report, f, indent=2, default=str)
        f.write('\n')


def config_from_real_cli():
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.entrypoints.cli.serve import ServeSubcommand
    from vllm.entrypoints.serve.utils.api_utils import cli_env_setup
    from vllm.usage.usage_lib import UsageContext
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    cli_env_setup()
    parser = FlexibleArgumentParser(prog='vllm')
    serve = ServeSubcommand()
    serve.subparser_init(parser.add_subparsers(dest='subparser', required=True))
    args = parser.parse_args(json.loads(os.environ['PM_REWIND_SERVE_ARGV']))
    serve.validate(args)
    if args.model_tag is not None:
        args.model = args.model_tag
    return AsyncEngineArgs.from_cli_args(args).create_engine_config(usage_context=UsageContext.OPENAI_API_SERVER)


def export_cpu_config(config, root):
    """Workers get the API's real config with the executor class reduced to metadata."""
    import copy
    import hashlib
    import pickle
    cloned = copy.deepcopy(config)
    executor = cloned.parallel_config.distributed_executor_backend
    cloned.parallel_config._pm_cpu_executor_supports_async = executor.supports_async_scheduling()
    cloned.parallel_config.distributed_executor_backend = 'ray'
    if config.compute_hash() != cloned.compute_hash():
        raise RuntimeError('Deferring executor binding changed the model compilation hash')
    data = pickle.dumps(cloned, protocol=pickle.HIGHEST_PROTOCOL)
    path = root / 'model-config.pkl'
    with path.open('xb') as f:
        f.write(data)
    return {'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
            'original_executor': f'{executor.__module__}.{executor.__qualname__}', 'cpu_executor': 'ray',
            'compute_hash': config.compute_hash()}


def load_cpu_config():
    import pickle
    from vllm.plugins import load_general_plugins
    load_general_plugins()
    with open(os.environ['PM_CPU_MODEL_CONFIG'], 'rb') as f:
        return pickle.load(f)


def meta_dispatch(op, dtype):
    """Real registered fake/Meta kernel of the attention custom op; never a TPU execution."""
    import torch
    values = {}
    for arg in op._schema.arguments:
        if arg.name == 'kv_cache':
            values[arg.name] = torch.empty((1, 1, 1, 1, 1), device='meta', dtype=dtype)
        elif arg.name == 'query':
            values[arg.name] = torch.empty((2, 1, 16), device='meta', dtype=dtype)
        elif str(arg.type) == 'Tensor':
            values[arg.name] = torch.empty((1,), device='meta', dtype=dtype)
        elif arg.has_default_value():
            values[arg.name] = arg.default_value
        elif str(arg.type).startswith('Optional['):
            values[arg.name] = None
        else:
            raise RuntimeError(f'unsupported metadata probe argument: {arg}')
    outputs = op(**values)
    assert len(outputs) == 2 and all(t.device.type == 'meta' for t in outputs)
    assert tuple(outputs[0].shape) == tuple(values['kv_cache'].shape)
    assert tuple(outputs[1].shape) == tuple(values['query'].shape)
    return {'schema': str(op._schema), 'shapes': [list(t.shape) for t in outputs],
            'scope': 'real registered Meta dispatch, not TPU execution'}


def tensor_specs(model):
    return [[name, list(shape), dtype, list(stride), kind]
            for name, shape, dtype, stride, kind in cpu_model.terminal_tensor_signature(model)]


def module_attrs(model):
    """Non-tensor module state, plus tensor attribute type/shape/dtype (never values or device)."""
    import torch
    simple = (int, float, bool, str, type(None), torch.dtype, torch.device)
    skip = {'_parameters', '_buffers', '_modules', '_non_persistent_buffers_set'}
    out = {}
    for name, module in model.named_modules():
        attrs = {}
        for key, value in vars(module).items():
            if key in skip or key.startswith(('_forward', '_backward', '_state_dict', '_load_state_dict')):
                continue
            if isinstance(value, simple) or (isinstance(value, tuple) and all(isinstance(v, simple) for v in value)):
                attrs[key] = repr(value)
            elif isinstance(value, torch.Tensor):
                attrs[key] = repr(('tensor', type(value).__qualname__, tuple(value.shape), str(value.dtype)))
            else:
                attrs[key] = 'type:' + type(value).__qualname__
        out[name] = attrs
    return out


def postload(model, config, rank):
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.distributed.parallel_state import cpu_model_parallel_geometry
    from vllm.model_executor.model_loader.utils import process_weights_after_loading
    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import set_vllm_model_wrapper_context
    started = time.perf_counter()
    with (cpu_model_parallel_geometry(tp_rank=rank, tp_size=TP), set_current_vllm_config(config),
          set_vllm_model_wrapper_context(mesh=None, vllm_config=config)):
        process_weights_after_loading(model, config.model_config, torch.device('cpu'))
    return time.perf_counter() - started


def scale_slots(model):
    from vllm.model_executor.layers.attention.attention import Attention
    return [[name, attr, str(t.device), str(t.dtype), list(t.shape), None if t.is_meta else float(t)]
            for name, layer in model.named_modules() if isinstance(layer, Attention)
            for attr in ('_k_scale_cpu', '_v_scale_cpu') for t in (getattr(layer, attr),)]


def role_groups(holder):
    import collections
    import re
    groups = collections.Counter(
        (r['role'], re.sub(r'\.\d+\.', '.N.', r['path']), tuple(r['shape']), r['dtype'])
        for r in holder.tensor_inventory if r['role'] not in ('live', 'reload_metadata'))
    return [{'role': k[0], 'path': k[1], 'shape': list(k[2]), 'dtype': k[3], 'count': n}
            for k, n in sorted(groups.items())]


def rejected(label, fn):
    try:
        fn()
    except Exception as exc:
        return {'case': label, 'rejected': True, 'error': f'{type(exc).__name__}: {exc}'}
    return {'case': label, 'rejected': False, 'error': None}


def prepare():
    """pm_rewind factory: capture two real rank holders, then verify their reuse on resume."""
    proof_root = Path(os.environ['PM_CPU_MODEL_PROOF'])
    stages = []
    with factory._cpu_preparation():
        state, _ = checked(stages, 'existing_worker_cpu_prefix', lambda: factory._prepare('workers'))
        configs = {rank: checked(stages, f'load_cpu_config_{rank}', load_cpu_config) for rank in RANKS}
        holders = {rank: checked(stages, f'prepare_holder_{rank}',
                                 lambda rank=rank: cpu_model.prepare_cpu_model(configs[rank], rank, TP))
                   for rank in RANKS}
        cpu_model.install_prepared_models(holders)
        import torch
        op0 = torch.ops.pallas.rpa_kernel_0.default
        dispatch = checked(stages, 'meta_dispatch', lambda: meta_dispatch(op0, configs[0].model_config.dtype))
        terminal = {rank: cpu_model.terminal_tensors(h.model) for rank, h in holders.items()}
        cpu_objects = {'pallas_0': op0}
        for rank, holder in holders.items():
            cpu_objects.update({f'holder:{rank}': holder, f'model:{rank}': holder.model,
                                f'config:{rank}': holder.config})
            cpu_objects.update({f'tensor:{rank}:{name}': t for name, t in terminal[rank].items()})
        captured = {rank: {
            'model_id': id(h.model), 'config_id': id(h.config), 'state': h.state,
            'tensor_count': len(terminal[rank]),
            'tensor_impls': {name: t._cdata for name, t in terminal[rank].items()},
            'roles': {role: sum(r['role'] == role for r in h.tensor_inventory) for role in LIFECYCLE_ROLES},
            'role_groups': role_groups(h), 'meta_bytes': h.device_bytes['meta'],
            'pallas': cpu_model.inspect_pallas_bindings(h), 'terminal_specs': tensor_specs(h.model),
            'compile_prefixes': list(h.compile_prefixes), 'holder_compile_counter': repr(h.compile_counter),
        } for rank, h in holders.items()}
        process_counter = repr(cpu_model._compile_prefix_counter_cell().cell_contents)
        if set(captured[RANKS[0]]['tensor_impls'].values()) & set(captured[RANKS[1]]['tensor_impls'].values()):
            raise RuntimeError('rank holders share terminal TensorImpls')
        if process_counter != 'count(0)' or any(row['compile_prefixes'] != ['_tpu_instance_0']
                                                for row in captured.values()):
            raise RuntimeError('holder compile_prefix differs from a fresh single-model worker')
        workers.seal_preloaded(state)
        workers.check_preloaded(state)
        factory._remember(state, 'workers', cpu_objects)
        capture = factory.get_prepared_state()
        capture.update(holders=captured, stages=stages, metadata_dispatch=dispatch, runtime_state=runtime_state(),
                       process_compile_counter=process_counter, scope=SCOPE)
        write_report(proof_root / 'constructed-once.json', capture)

    def resume(argv):
        if list(argv) != ['check']:
            raise ValueError('native holder probe accepts only check')
        workers.apply_prepared_environment(state)
        workers.check_preloaded(state)
        proof = factory.get_prepared_state()
        assert proof['pid'] == proof['proc_pid'] != capture['capture_pid']
        assert proof['saved_cpu_ids'] == proof['current_cpu_ids'] == capture['saved_cpu_ids']
        assert proof['saved_module_ids'] == proof['current_module_ids'] == capture['saved_module_ids']
        assert proof['native_address'] == capture['saved_native_address']
        assert proof['native_bytes'] == capture['saved_native_bytes']
        for rank, h in holders.items():
            now = cpu_model.terminal_tensors(h.model)
            assert id(h.model) == captured[rank]['model_id'] and h.state == 'prepared'
            assert {name: t._cdata for name, t in now.items()} == captured[rank]['tensor_impls']
            assert all(t.is_meta for t in now.values())
            assert cpu_model.inspect_pallas_bindings(h) == captured[rank]['pallas']
        assert meta_dispatch(op0, configs[0].model_config.dtype) == dispatch
        workers.assert_cpu_boundary(ray_free=True, loader_unloaded=True)

        import torch
        from vllm.model_executor.model_loader.utils import initialize_model
        target = RANKS[1]
        fresh = load_cpu_config()
        changed = load_cpu_config()
        changed.scheduler_config.max_num_seqs += 1
        negatives = [
            rejected('rank_without_holder', lambda: cpu_model.claim_prepared_cpu_model(fresh, tp_rank=3, tp_size=TP)),
            rejected('wrong_tp_size', lambda: cpu_model.claim_prepared_cpu_model(fresh, tp_rank=0, tp_size=4)),
            rejected('holder_rank_mismatch', lambda: cpu_model.validate_prepared_model(holders[0], fresh, target, TP)),
            rejected('changed_config', lambda: cpu_model.validate_prepared_model(holders[target], changed, target, TP)),
            rejected('tpu_claim_without_distributed', lambda: cpu_model.claim_prepared_model(fresh)),
        ]
        with torch.device('meta'):
            negatives.append(rejected('initialize_model_outside_real_loader', lambda: initialize_model(
                vllm_config=fresh, model_config=fresh.model_config)))
        assert all(row['rejected'] for row in negatives), negatives
        assert all(h.state == 'prepared' for h in holders.values())
        counter_before = repr(cpu_model._compile_prefix_counter_cell().cell_contents)
        assert counter_before == 'count(0)', counter_before
        workers.assert_cpu_boundary(ray_free=True, loader_unloaded=True)

        holder = cpu_model.claim_prepared_cpu_model(fresh, tp_rank=target, tp_size=TP)
        before = {name: (id(t), type(t), dict(t.__dict__)) for name, t in cpu_model.terminal_tensors(holder.model).items()}
        started = time.perf_counter()
        model = cpu_model.materialize_prepared_model(holder, fresh, 'cpu')
        materialize_s = time.perf_counter() - started
        after = cpu_model.terminal_tensors(model)
        assert model is holder.model and id(model) == captured[target]['model_id']
        assert {name: id(t) for name, t in after.items()} == {name: row[0] for name, row in before.items()}
        assert all(type(t) is before[name][1] and all(t.__dict__.get(k) is v for k, v in before[name][2].items())
                   for name, t in after.items())
        assert all(t.device.type == 'cpu' for t in after.values())
        assert tensor_specs(model) == captured[target]['terminal_specs']
        assert all(fresh.compilation_config.static_forward_context[name] is layer
                   for name, layer in holder.config.compilation_config.static_forward_context.items())
        assert holders[RANKS[0]].state == 'prepared'
        assert all(t.is_meta for t in cpu_model.terminal_tensors(holders[RANKS[0]].model).values())
        slots_before = scale_slots(model)
        postload_s = postload(model, fresh, target)
        workers.assert_cpu_boundary(ray_free=True, loader_unloaded=True)
        proof.update(
            success=True, rebuilt=False, target_rank=target, negatives=negatives,
            materialization=holder.materialization, materialize_s=materialize_s, postload_s=postload_s,
            materialized_bytes=sum(t.numel() * t.element_size() for t in after.values()),
            terminal_specs_before_postload=captured[target]['terminal_specs'],
            terminal_specs_after_postload=tensor_specs(model), module_attrs_after_postload=module_attrs(model),
            scale_slots_before_postload=slots_before, scale_slots_after_postload=scale_slots(model),
            other_holder_state=holders[RANKS[0]].state, holders_capture=captured,
            metadata_dispatch=dispatch, runtime_state=runtime_state(), scope=SCOPE)
        write_report(proof_root / 'native-restored.json', proof)
        print('CPU_MODEL_HOLDERS_PROOF ' + json.dumps({key: proof[key] for key in (
            'success', 'rebuilt', 'capture_pid', 'pid', 'target_rank', 'materialize_s', 'postload_s',
            'materialized_bytes', 'runtime_state')}), flush=True)
        return 0

    inputs = tuple(str(p) for p in sorted(Path(configs[0].model_config.model).iterdir())
                   if p.suffix in ('.json', '.model', '.txt', '.jinja'))
    resume.rewind_input_files = (*inputs, os.environ['PM_CPU_MODEL_CONFIG'])
    return resume


def main():
    mode, root = sys.argv[1], Path(sys.argv[2])
    report, stages = {'mode': mode, 'pid': os.getpid(), 'stages': []}, []
    report['stages'] = stages
    try:
        with factory._cpu_preparation():
            if mode == 'export':
                checked(stages, 'existing_api_cpu_prefix', lambda: factory._prepare('api'), ray_free=False)
                config = checked(stages, 'real_cli_config', config_from_real_cli, ray_free=False)
                report['config_export'] = checked(stages, 'export_cpu_config',
                                                  lambda: export_cpu_config(config, root), ray_free=False)
            elif mode == 'reference':
                rank = int(sys.argv[3])
                report['rank'] = rank
                checked(stages, 'existing_worker_cpu_prefix', lambda: factory._prepare('workers'))
                config = checked(stages, 'load_cpu_config', load_cpu_config)
                model = checked(stages, 'reference_construction',
                                lambda: cpu_model.build_cpu_reference_model(config, rank, TP))
                report.update(terminal_specs_before_postload=tensor_specs(model),
                              scale_slots_before_postload=scale_slots(model))
                report['postload_s'] = checked(stages, 'real_postload', lambda: postload(model, config, rank))
                report.update(terminal_specs_after_postload=tensor_specs(model),
                              module_attrs_after_postload=module_attrs(model),
                              scale_slots_after_postload=scale_slots(model))
            else:
                raise ValueError(mode)
        report['outcome'] = 'passed'
    except Exception as exc:
        report.update(outcome='blocked', error=repr(exc), traceback=traceback.format_exc())
    finally:
        report['final_state'] = runtime_state()
        name = f'reference-rank{report["rank"]}.json' if 'rank' in report else f'{mode}.json'
        write_report(root / name, report)
        print('CPU_MODEL_PROBE_RESULT ' + json.dumps({'mode': mode, 'outcome': report['outcome'],
                                                      'error': report.get('error')}), flush=True)
    return 0 if report['outcome'] == 'passed' else 1


if __name__ == '__main__':
    sys.exit(main())
