#!/usr/bin/env python3
"""Code-size accounting for the TPUFlex port to vLLM-TorchTPU: code lines (no blanks, comments or
docstrings) per mechanism, from the top-level functions/classes of each file plus the vLLM source patches.

Usage: pm_loc.py [--json OUT]   (run from the repo root or scripts/)
"""
import ast
import io
import json
import pathlib
import sys
import tokenize

HERE = pathlib.Path(__file__).resolve().parent

# mechanism -> [(file, [top-level names] or None for the whole file)]
GROUPS = {
    'pre-spawn gate (Decompose)': [('pm_torchtpu_plugin.py', ['_patch_worker_wrapper', 'register', '_log'])],
    'terminal weight snapshot + async restore (Cache, Overlap)': [
        ('pm_torchtpu_plugin.py', ['_patch_worker_class', '_patch_snapshot', '_rank_world', '_tensors', '_paths',
                                   '_snapshot_ready', '_save', '_restore', '_join_async'])],
    'zygote: pre-imported fork server (Decompose, Overlap)': [
        ('pm_zygote.py', None),
        ('pm_torchtpu_plugin.py', ['ZygoteProcess', 'ZygoteContext', '_install_zygote_client'])],
    'pre-TPU CPU-image controller (experimental Cache; zygote integration counted above)': [
        ('pm_rewind.py', None)],
    'Ray CPU-prefix factory and native worker/API continuation (experimental Cache)': [
        ('pm_ray_zygote.py', None), ('pm_rewind_ray.py', None)],
    'DMTCP real-PID/thread and exec rebind (native source patch)': [
        ('pm_rewind_dmtcp_pid.patch', None)],
    'real TorchTPU registration after final worker binding (vLLM source patch)': [
        ('pm_source_patches.py:rewind_deferred_tpu_init', None)],
    'prepared API CPU objects and real engine continuation (vLLM source patches)': [
        ('pm_source_patches.py:prepare_cpu_constructor', None),
        ('pm_source_patches.py:prepare_cpu_resources', None),
        ('pm_source_patches.py:prepare_cpu_stat_plugins', None),
        ('pm_source_patches.py:prepare_cpu_renderer', None),
        ('pm_source_patches.py:prepare_cpu_resume', None),
        ('pm_source_patches.py:prepare_cpu_from_config', None),
        ('pm_source_patches.py:prepare_cpu_from_config_forward', None),
        ('pm_source_patches.py:prepare_cpu_shutdown', None),
        ('pm_source_patches.py:prepare_cpu_launcher_handoff', None),
        ('pm_source_patches.py:prepare_cpu_launcher_resume', None)],
    'native bundle verification and TP8 experiment/ownership harness (not startup mechanism)': [
        ('pm_rewind_native.py', None), ('pm_rewind_tpu.py', None)],
    'renderer built during engine start (Overlap; vLLM source patch)': [('pm_source_patches.py:overlap_renderer', None)],
    'tokenizer/processor warm-up during worker boot (Overlap)': [
        ('pm_torchtpu_plugin.py', ['_install_warm_hooks', '_warm_in_thread'])],
    'compile-cache replay without deepcopy/concrete args (Cache; vLLM source patch)': [
        ('pm_source_patches.py:fast_replay', None)],
    'skip the tokenizer-loading multimodal probe under --language-model-only (vLLM source patch)': [
        ('pm_source_patches.py:skip_mm_probe', None)],
    'outer AOT artifact deserialized during the weight restore (Overlap; +4 lines in _patch_aot_timing, +2 in _patch_snapshot)': [
        ('pm_torchtpu_plugin.py', ['_preload_aot'])],
    'slim outer AOT artifact: no tensor meta on interior nodes (Cache; +4 lines in the init_device hook)': [
        ('pm_torchtpu_plugin.py', ['_patch_aot_slim', '_strip_meta'])],
    'bank executables deserialized during the weight restore (Overlap; +4 lines in hooks)': [
        ('pm_torchtpu_plugin.py', ['_preload_bank', '_install_bank_preload'])],
    'sampler bank: helper programs AOT-compiled once, loaded on later boots (Cache; +2 lines in the init_device hook)': [
        ('pm_torchtpu_plugin.py', ['_install_sampler_bank', '_sb_count', '_sb_guard_reason', '_sb_signature',
                                   '_sb_prune_guard_state', '_sb_load_or_build'])],
    'GCS artifact pack/upload/fetch (fresh host)': [
        ('pm_artifacts.py', ['split_gs', 'cmd_pack', 'cmd_upload', 'cmd_delete', 'Streamer', 'split_ranges',
                             'stream_into', 'gcs_token', 'gcs_bytes', 'gcs_json', '_View', 'extract', 'cmd_fetch',
                             'main'])],
    'diagnostics only (timing, profiling, sampler; not needed for speed)': [
        ('pm_torchtpu_plugin.py', ['_patch_bank', '_patch_aot_timing', '_install_profiles', '_profiled',
                                   '_start_sampler']),
        ('pm_rewind_probe.py', None), ('pm_rewind_ray_probe.py', None)],
    'fair-baseline tool (stock checkpoint via Run:ai; not TPUFlex)': [('pm_artifacts.py', ['gcs_list', 'cmd_fetch_ckpt'])],
}


def code_lines(src, lo=1, hi=None):
    """Line numbers in [lo, hi] that carry code (not blank, comment-only or a docstring)."""
    lines = set()
    doc = set()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(getattr(first, 'value', None), ast.Constant) \
                    and isinstance(first.value.value, str):
                doc.update(range(first.lineno, first.end_lineno + 1))
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
                        tokenize.ENDMARKER):
            continue
        for ln in range(tok.start[0], tok.end[0] + 1):
            if ln not in doc:
                lines.add(ln)
    hi = hi or len(src.splitlines())
    return {ln for ln in lines if lo <= ln <= hi}


def count(file, names):
    if ':' in file:  # a source patch: changed code lines of the replacement blocks
        sys.path.insert(0, str(HERE))
        import difflib

        import pm_source_patches as sp
        name = file.split(':')[1]
        add = rem = 0
        for p in sp.PATCHES:
            if p['name'] != name:
                continue
            for line in difflib.ndiff(p['old'].splitlines(), p['new'].splitlines()):
                body = line[2:].strip()
                if body and not body.startswith('#'):
                    add += line.startswith('+ ')
                    rem += line.startswith('- ')
        return add, f'vLLM source: +{add} -{rem}'
    src = (HERE / file).read_text()
    if file.endswith('.patch'):
        add = rem = 0
        for line in src.splitlines():
            if line.startswith(('+++', '---')) or line[:1] not in ('+', '-'):
                continue
            body = line[1:].strip()
            if body and not body.startswith('//'):
                add += line.startswith('+')
                rem += line.startswith('-')
        return add, f'DMTCP native source: +{add} -{rem}'
    if names is None:
        return len(code_lines(src)), file
    tree = ast.parse(src)
    spans = {n.name: (n.lineno, n.end_lineno) for n in tree.body
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    total = 0
    for n in names:
        lo, hi = spans[n]
        total += len(code_lines(src, lo, hi))
    return total, f'{file}: {", ".join(names)}'


def main():
    out = {}
    for group, parts in GROUPS.items():
        n = 0
        where = []
        for file, names in parts:
            c, w = count(file, names)
            n += c
            where.append(f'{w} ({c})')
        out[group] = {'code_lines': n, 'where': where}
        print(f'{n:5d}  {group}\n       ' + '\n       '.join(where))
    if '--json' in sys.argv:
        pathlib.Path(sys.argv[sys.argv.index('--json') + 1]).write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
