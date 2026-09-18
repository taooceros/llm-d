#!/usr/bin/env python3
"""Export historical research evidence without executing report or accelerator code.

Python stdlib only. All metric projections are exact copies of recorded JSON values.
Archives contain original research data (including token IDs), not executable code
qualification. Re-running with the same inputs produces byte-identical assets.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile

REPORT = 'results/hetero/reports/fp32_state_20260918_v2'
BF16 = '6ffbc8d3fe92a6f2a1f410d7b30a7c3fb2cddddc6c657a61f0cc666a06da7186'
LIMIT = 850_000
PART_BYTES = 900_000_000
SECRET_PATTERNS = {
    'private_key': rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----',
    'github_token': rb'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})\b',
    'google_api_key': rb'\bAIza[A-Za-z0-9_-]{35}\b',
    'google_oauth_token': rb'\bya29\.[A-Za-z0-9_-]{30,}',
    'aws_access_key': rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
    'huggingface_token': rb'\bhf_[A-Za-z0-9]{30,}\b',
    'credential_assignment': rb'(?i)["\x27]?(?:api_key|client_secret|access_token|refresh_token|password)["\x27]?\s*[:=]\s*["\x27][A-Za-z0-9_./+@=-]{20,}["\x27]',
}
PATTERNS = {k: re.compile(v) for k, v in SECRET_PATTERNS.items()}
PREFIXES = {
    'private_key': (b'private key',),
    'github_token': (b'ghp_', b'gho_', b'ghu_', b'ghs_', b'ghr_', b'github_pat_'),
    'google_api_key': (b'aiza',), 'google_oauth_token': (b'ya29.',),
    'aws_access_key': (b'akia', b'asia'), 'huggingface_token': (b'hf_',),
    'credential_assignment': (b'api_key', b'client_secret', b'access_token', b'refresh_token', b'password'),
}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False) + '\n').encode()


def write_json(path, value):
    raw = encode(value)
    if len(raw) >= 1_000_000:
        raise ValueError(f'Git evidence file exceeds 1 MB: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def shards(out, prefix, values):
    names, part, size = [], [], 2
    for value in values:
        length = len(encode(value))
        if length >= LIMIT:
            raise ValueError(f'Single {prefix} record too large')
        if part and size + length > LIMIT:
            name = f'{prefix}-{len(names) + 1:03d}.json'
            write_json(out / name, part)
            names.append(name)
            part, size = [], 2
        part.append(value)
        size += length
    if part:
        name = f'{prefix}-{len(names) + 1:03d}.json'
        write_json(out / name, part)
        names.append(name)
    return names


def read_shards(out, names):
    return [record for name in names for record in json.loads((out / name).read_text())]


def scalar_projection(record):
    return {k: v for k, v in record.items() if not isinstance(v, (dict, list))}


def run_summary(run, index):
    result = scalar_projection(run)
    for key in ('eligibility', 'initialization_provenance', 'iteration_evidence',
                'transfer_accounting', 'migration_coverage', 'counters',
                'protocol_exclusions', 'initial_placement_counts'):
        if key in run:
            result[key] = run[key]
    result['source_pointer'] = f'/arms/{index}'
    return result


def exclusion_reason(rel):
    p = PurePosixPath(rel)
    if p.is_absolute() or '..' in p.parts or '\\' in rel:
        return 'unsafe relative path'
    if any(x in {'.git', '.jj', '.omp', '.ssh', '.config', '__pycache__',
                 'node_modules', '.venv', 'target'} for x in p.parts):
        return 'local tooling, context, credentials, or generated binaries'
    name = p.name.lower()
    if name in {'agents.md', '.env', '.netrc', '.git-credentials', 'credentials',
                'credentials.json', 'application_default_credentials.json',
                'id_rsa', 'id_ed25519'} or name.endswith(('.pem', '.key', '.pyc')):
        return 'credential/context/generated file name'
    return None


def scan_hash(path):
    sha, found, tail = hashlib.sha256(), set(), b''
    with path.open('rb') as stream:
        while block := stream.read(1024 * 1024):
            sha.update(block)
            window = tail + block
            lower = window.lower()
            for label, pattern in PATTERNS.items():
                if any(prefix in lower for prefix in PREFIXES[label]) and pattern.search(window):
                    found.add(label)
            tail = window[-4096:]
    return sha.hexdigest(), sorted(found)


def export_summaries(root, out, migration):
    source = root / REPORT / 'RESULTS.json'
    results = json.loads(source.read_text())
    selected = [(i, r) for i, r in enumerate(results['arms']) if r['eligibility']['headline_eligible']]
    original = [(i, r) for i, r in enumerate(results['arms']) if r.get('predeclaration_sha256') == BF16]
    if len(selected) != 102 or len(original) != 39:
        raise ValueError('Expected original 102 FP32 and 39 BF16 cohort records')
    generated = {}
    for name in ('REPORT.md', 'goal_audit.json'):
        source_file = root / REPORT / name
        _, findings = scan_hash(source_file)
        if findings:
            raise ValueError(f'Credential pattern in curated {name}: {findings}')
        shutil.copyfile(source_file, out / name)
    generated['fp32_runs'] = shards(out, 'fp32-runs', [run_summary(r, i) for i, r in selected])
    generated['bf16_runs'] = shards(out, 'historical-bf16-runs', [run_summary(r, i) for i, r in original])
    trials = results['mechanism_cost']['report']['trials']
    if len(trials) != 18:
        raise ValueError('Expected 18 original mechanism trials')
    projected_trials = []
    for i, trial in enumerate(trials):
        projected = scalar_projection(trial)
        for key in ('metrics', 'errors', 'passed_flags', 'useful_token_audit', 'artifact'):
            if key in trial:
                projected[key] = trial[key]
        projected['source_pointer'] = f'/mechanism_cost/report/trials/{i}'
        projected_trials.append(projected)
    generated['mechanism_trials'] = shards(out, 'mechanism-trials', projected_trials)
    contrasts = {key: results[key] for key in ('criteria', 'contrast_completion', 'mechanism_cost_summary')}
    contrasts['efficiency_contrasts'] = results['efficiency_analysis']['contrasts']
    contrasts['mechanism_pairs'] = results['mechanism_cost']['report']['pairs']
    write_json(out / 'paired-contrasts.json', contrasts)
    diagnostics = {'retained_history': results['retained_history'],
                   'historical_diagnostics': results['goal_audit']['historical_diagnostics'],
                   'excluded_runs': [run_summary(r, i) for i, r in enumerate(results['arms'])
                                     if not r['eligibility']['headline_eligible']],
                   'direct_ray_diagnostics': results['direct_ray_diagnostics'],
                   'qualification_validation': results['qualification_validation'],
                   'missing_or_excluded_scheduled_runs': results['missing_or_excluded_scheduled_runs']}
    write_json(out / 'historical-diagnostics.json', diagnostics)
    freeze = results['predeclaration']['source_freeze']
    comparisons = []
    for path, expected in sorted(freeze['source_sha256'].items()):
        original_path = root / path
        observed = digest(original_path) if original_path.is_file() else None
        comparisons.append({'path': path, 'frozen_sha256': expected, 'current_workspace_sha256': observed,
                            'matches_frozen': expected == observed})
    provenance = {'source_results': {'path': str(source.relative_to(root)), 'sha256': digest(source),
                                    'size': source.stat().st_size},
                  'migration_source_commit': migration['source_commit'],
                  'predeclaration': results['predeclaration'], 'source_comparison': comparisons,
                  'historical_bf16_cohort': BF16,
                  'scope': 'Historical measurements on the original frozen runtime; no native upstream port has been physically requalified.',
                  'archival_inputs': 'Original research data, including dataset manifest token IDs; not code.',
                  'summary_files': generated}
    write_json(out / 'provenance.json', provenance)
    checks = {
        'report_exact_bytes': (out / 'REPORT.md').read_bytes() == (root / REPORT / 'REPORT.md').read_bytes(),
        'goal_audit_exact_bytes': (out / 'goal_audit.json').read_bytes() == (root / REPORT / 'goal_audit.json').read_bytes(),
        'fp32_projection_exact': read_shards(out, generated['fp32_runs']) == [run_summary(r, i) for i, r in selected],
        'bf16_projection_exact': read_shards(out, generated['bf16_runs']) == [run_summary(r, i) for i, r in original],
        'mechanism_projection_exact': read_shards(out, generated['mechanism_trials']) == projected_trials,
        'paired_contrasts_exact': json.loads((out / 'paired-contrasts.json').read_text()) == contrasts,
        'historical_diagnostics_exact': json.loads((out / 'historical-diagnostics.json').read_text()) == diagnostics,
    }
    if not all(checks.values()):
        raise ValueError('Summary verification failed')
    return {'checks': checks, 'fp32_runs': len(selected), 'historical_bf16_runs': len(original),
            'mechanism_trials': len(trials), 'useful_tokens': sum(r['useful_output_tokens'] for _, r in selected),
            'iteration_rows': sum(r['iteration_evidence']['rows'] for _, r in selected),
            'source_freeze_mismatches': [r for r in comparisons if not r['matches_frozen']]}


def build_archives(root, out, release, migration, native):
    paths = {r['path'] for r in migration['retained_outside_source_copy']}
    local_exclusions = {r['path']: r['reason'] for r in migration['retained_outside_source_copy']
                        if r['reason'].startswith('local tooling')}
    for dirname in ('results', 'experiments', 'data'):
        paths.update(str(p.relative_to(root)) for p in (root / dirname).rglob('*') if p.is_file() or p.is_symlink())
    from audit_publication import assess
    publication = assess(root, {p for p in paths if p not in local_exclusions}, PATTERNS, PREFIXES)
    write_json(out / 'publication-audit.json', publication)
    publication_exclusions = {row['path']: row['reason'] for row in publication['exclusions']}
    excluded, records, findings = [], [], []
    for rel in sorted(paths):
        path = root / rel
        reason = local_exclusions.get(rel) or publication_exclusions.get(rel) or exclusion_reason(rel)
        if path.is_symlink() or (path.exists() and not path.is_file()):
            reason = 'non-regular file'
        if not path.exists():
            reason = 'missing original input'
        if reason:
            record = {'path': rel, 'reason': reason}
            if rel in publication_exclusions and path.is_file():
                record.update({'sha256': digest(path), 'size': path.stat().st_size})
            excluded.append(record)
            continue
        sha, matches = scan_hash(path)
        if matches:
            findings.append({'path': rel, 'patterns': matches})
            excluded.append({'path': rel, 'reason': 'obvious credential pattern; original not modified'})
            continue
        records.append({'path': rel, 'sha256': sha, 'size': path.stat().st_size})
    native_metadata = json.loads((native / 'commits.json').read_text())
    native_paths = ['commits.json'] + [f'{repo}/{entry["path"]}' for repo, entry in native_metadata.items()]
    for rel in sorted(native_paths):
        path = native / rel
        if exclusion_reason(rel) or path.is_symlink() or not path.is_file():
            raise ValueError(f'Unsafe native provenance path: {rel}')
        sha, matches = scan_hash(path)
        if matches:
            raise ValueError(f'Credential pattern in native provenance: {rel}: {matches}')
        if rel != 'commits.json' and sha != native_metadata[rel.split('/')[0]]['sha256']:
            raise ValueError(f'Native provenance hash mismatch: {rel}')
        records.append({'path': f'native-provenance/{rel}', 'sha256': sha, 'size': path.stat().st_size})
    # Partition by uncompressed size: even incompressible gzip stays far below 2 GB.
    groups = {}
    for record in records:
        path = record['path']
        group = ('native-provenance' if path.startswith('native-provenance/') else
                 'hetero' if path.startswith('results/hetero/') else
                 'prior-results' if path.startswith('results/') else 'research-inputs')
        groups.setdefault(group, []).append(record)
    assets, inventory = [], []
    for group, group_records in sorted(groups.items()):
        parts, current, total = [], [], 0
        for record in group_records:
            if record['size'] > PART_BYTES:
                raise ValueError(f'Single input too large for archive partition: {record["path"]}')
            if current and total + record['size'] > PART_BYTES:
                parts.append(current)
                current, total = [], 0
            current.append(record)
            total += record['size']
        if current:
            parts.append(current)
        for number, part in enumerate(parts, 1):
            name = f'{group}-{number:03d}.tar.gz'
            destination = release / name
            with destination.open('wb') as raw:
                with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0, compresslevel=6) as compressed:
                    with tarfile.open(fileobj=compressed, mode='w|', format=tarfile.PAX_FORMAT) as archive:
                        for record in part:
                            path = (native / record['path'].removeprefix('native-provenance/')
                                    if record['path'].startswith('native-provenance/') else root / record['path'])
                            info = tarfile.TarInfo(record['path'])
                            info.size, info.mode, info.mtime = record['size'], 0o644, 0
                            with path.open('rb') as source:
                                archive.addfile(info, source)
                            inventory.append(dict(record, asset=name))
            size = destination.stat().st_size
            if size >= 2_000_000_000:
                raise ValueError(f'Asset exceeds conservative GitHub limit: {name}')
            assets.append({'name': name, 'sha256': digest(destination), 'size': size,
                           'files': len(part), 'uncompressed_bytes': sum(r['size'] for r in part)})
    inventory_names = shards(out, 'archive-inventory', inventory)
    index = {'format': 1, 'repository': 'taooceros/llm-d', 'release_tag': 'research-evidence-20260918',
             'classification': 'Original research data and archival artifact inputs, including token IDs; not code.',
             'restore_paths': 'Original relative workspace paths; native provenance additionally restores under native-provenance/. Safe overlay requires --merge.',
             'native_provenance': native_metadata,
             'assets': assets,
             'inventories': [{'name': name, 'sha256': digest(out / name), 'size': (out / name).stat().st_size}
                             for name in inventory_names],
             'files': len(inventory), 'uncompressed_bytes': sum(r['size'] for r in inventory),
             'exclusions': excluded,
             'encoded_input_assessment': {'name': 'publication-audit.json',
                                          'sha256': digest(out / 'publication-audit.json'),
                                          'size': (out / 'publication-audit.json').stat().st_size},
             'public_raw_reproduction': 'Affected reconstructable originals are withheld, not redacted. Complete historical raw-audit regeneration requires the original private inputs. Public compact metrics and frozen provenance remain exact.',
             'credential_scan': {'patterns': sorted(PATTERNS), 'findings': findings,
                                 'scope': 'Raw bytes and decoded gzip content assessed; corpus source-ID and affected token-prefix overlap assessed in publication-audit.json. Unmapped token arrays were not decoded and are not claimed credential-free.'}}
    write_json(out / 'asset-index.json', index)
    for name in inventory_names + ['asset-index.json', 'publication-audit.json']:
        shutil.copyfile(out / name, release / name)
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path('/workspace'))
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--release-dir', type=Path, default=Path('/tmp/hetero-research-release'))
    parser.add_argument('--migration-index', type=Path, default=Path(__file__).resolve().parent.parent / 'migration-index.json')
    parser.add_argument('--native-provenance', type=Path, default=Path('/tmp/native-research-provenance'))
    args = parser.parse_args()
    root, out, release = args.workspace.resolve(), args.output.resolve(), args.release_dir.resolve()
    if root == out or root in out.parents or root == release or root in release.parents:
        raise ValueError('Exporter must never write into original workspace')
    out.mkdir(parents=True, exist_ok=True)
    release.mkdir(parents=True, exist_ok=True)
    migration = json.loads(args.migration_index.read_text())
    verification = export_summaries(root, out, migration)
    index = build_archives(root, out, release, migration, args.native_provenance.resolve())
    verification.update({'archived_files': index['files'], 'archived_original_bytes': index['uncompressed_bytes'],
                         'compressed_bytes': sum(a['size'] for a in index['assets']),
                         'assets': index['assets'], 'exclusions': index['exclusions'],
                         'credential_scan': index['credential_scan'],
                         'metrics_recomputed': False, 'benchmarks_executed': False,
                         'missing_data_policy': 'Original nulls, missing fields, statuses and exclusions are retained; no metric is simulated.'})
    write_json(out / 'verification.json', verification)
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
