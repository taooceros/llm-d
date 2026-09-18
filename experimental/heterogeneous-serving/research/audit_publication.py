#!/usr/bin/env python3
"""Assess corpus credential-record overlap without displaying prompts or tokens.

Credential-shaped data are conservatively withheld even when possibly examples.
Token arrays are not declared credential-free: matching source identifiers and
known affected prompt prefixes is only an overlap assessment, not full decoding.
"""
import argparse
import collections
import gzip
import json
from pathlib import Path


def assess(root, paths, patterns, prefixes):
    corpus_path = root / 'data/workloads/sharegpt_large.json'
    corpus = json.loads(corpus_path.read_text())
    corpus_ids = {row['id'] for row in corpus}
    flagged = {}
    for row in corpus:
        raw = row['prompt'].encode()
        lower = raw.lower()
        hits = [name for name, pattern in patterns.items()
                if any(prefix in lower for prefix in prefixes[name]) and pattern.search(raw)]
        if hits:
            flagged[row['id']] = hits
    del corpus
    manifest_coverage, bad_prefixes, clean_prefixes, affected_requests = [], set(), set(), set()
    # Include current and historical copies; only source IDs are disclosed.
    manifests = sorted(set((root / 'data/manifests').glob('*.json')) |
                       set((root / 'results/hetero/manifests_diagnostic_v1').glob('*.json')) |
                       set((root / 'results/hetero/artifacts/manifests').glob('*.json')))
    for path in manifests:
        data = json.loads(path.read_text())
        rows = data.get('requests', [])
        overlap = []
        for row in rows:
            source_id = row.get('source_id')
            tokens = row.get('prompt_token_ids', [])
            prefix = tuple(tokens[:32])
            if source_id in flagged:
                overlap.append({'source_id': source_id, 'request_id': row.get('request_id')})
                if len(prefix) == 32:
                    bad_prefixes.add(prefix)
                affected_requests.add(row['request_id'])
            elif source_id in corpus_ids and len(prefix) == 32:
                clean_prefixes.add(prefix)
        manifest_coverage.append({'path': str(path.relative_to(root)), 'requests': len(rows),
                                  'mapped_source_ids': sum(row.get('source_id') in corpus_ids for row in rows),
                                  'overlap_count': len(overlap), 'overlap_identifiers': overlap})
    exclusions, token_files, parse_errors = [], [], []
    totals = collections.Counter()
    for rel in sorted(paths):
        path = root / rel
        if not path.is_file() or path.is_symlink():
            continue
        totals['files_assessed'] += 1
        if rel == 'data/workloads/sharegpt_large.json':
            exclusions.append({'path': rel, 'reason': 'raw corpus contains credential-pattern records'})
            continue
        compressed = path.suffix == '.gz'
        opener = gzip.open if compressed else open
        with opener(path, 'rb') as stream:
            raw = stream.read()
        if compressed:
            totals['gzip_files_decompressed'] += 1
        lower = raw.lower()
        reasons = set()
        for name, pattern in patterns.items():
            if any(prefix in lower for prefix in prefixes[name]) and pattern.search(raw):
                reasons.add('credential pattern in decoded file: ' + name)
        suffix = Path(path.stem).suffix if compressed else path.suffix
        counts = collections.Counter()
        def visit(value, affected=False):
            if isinstance(value, dict):
                request = value.get('request_id', '')
                affected = (affected or
                            (isinstance(request, str) and any(request.endswith(x) for x in affected_requests)) or
                            (isinstance(value.get('source_id'), str) and value['source_id'] in flagged) or
                            (isinstance(value.get('id'), str) and value['id'] in flagged) or
                            any(source_id in flagged for source_id in (value.get('source_ids') or []) if isinstance(source_id, str)))
                for key, item in value.items():
                    if isinstance(item, list) and 'token' in key.lower() and item and all(isinstance(x, int) for x in item[:32]):
                        counts['token_arrays'] += 1
                        prefix = tuple(item[:32])
                        if len(prefix) == 32 and prefix in bad_prefixes:
                            counts['affected_prompt_prefix_matches'] += 1
                            reasons.add('contains affected encoded prompt prefix')
                        elif len(prefix) == 32 and prefix in clean_prefixes:
                            counts['clean_manifest_prompt_prefix_matches'] += 1
                        else:
                            counts['token_arrays_without_source_mapping'] += 1
                        if affected:
                            reasons.add('contains encoded tokens associated with affected request')
                    elif isinstance(item, (dict, list)):
                        visit(item, affected)
                    elif affected and key in {'text', 'prompt', 'output_text'} and item:
                        reasons.add('contains text associated with affected request')
                    elif affected and key in {'token_id', 'output_token_id'} and isinstance(item, int):
                        reasons.add('contains encoded tokens associated with affected request')
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, (dict, list)):
                        visit(item, affected)
        try:
            if suffix == '.json':
                visit(json.loads(raw))
            elif suffix == '.jsonl':
                for line in raw.splitlines():
                    if line.strip():
                        visit(json.loads(line))
        except (ValueError, UnicodeDecodeError) as error:
            parse_errors.append({'path': rel, 'error_type': type(error).__name__})
            reasons.add('JSON content could not be assessed')
        if counts:
            token_files.append({'path': rel, **counts})
            totals.update(counts)
        if reasons:
            exclusions.append({'path': rel, 'reason': '; '.join(sorted(reasons))})
    receipt_path = Path(__file__).with_name('tokenizer-credential-check.json')
    receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else None
    return {'format': 1, 'corpus_records': len(corpus_ids), 'flagged_corpus_record_count': len(flagged),
            'flagged_source_identifiers': [{'source_id': key, 'patterns': value} for key, value in sorted(flagged.items())],
            'manifest_coverage': manifest_coverage, 'totals': dict(totals), 'exclusions': exclusions,
            'parse_errors': parse_errors, 'token_file_assessments': token_files,
            'tokenizer_decoding': {'scope': 'Only the affected W1 prompt decoded using the exact tokenizer on CPU; other encoded arrays not decoded.', 'receipt': receipt},
            'coverage_limit': 'All selected gzip files decompressed before scanning. Known affected corpus IDs, request IDs and 32-token prompt prefixes conservatively withheld. Other token arrays, especially generated output, were not decoded; numeric regex absence is NOT evidence of credential freedom.',
            'metrics_policy': 'Original metrics and provenance are never changed; reconstructable affected raw files are excluded and retained only in the unchanged original workspace.'}


def main():
    import export_evidence as exporter
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path('/workspace'))
    parser.add_argument('--migration-index', type=Path, default=Path(__file__).resolve().parent.parent / 'migration-index.json')
    parser.add_argument('--output', type=Path, default=Path('/tmp/hetero-research-release/publication-audit.json'))
    args = parser.parse_args()
    migration = json.loads(args.migration_index.read_text())
    paths = {row['path'] for row in migration['retained_outside_source_copy'] if row['reason'].startswith('research')}
    for dirname in ('results', 'experiments', 'data'):
        paths.update(str(path.relative_to(args.workspace)) for path in (args.workspace / dirname).rglob('*') if path.is_file())
    result = assess(args.workspace, paths, exporter.PATTERNS, exporter.PREFIXES)
    args.output.write_text(json.dumps(result, sort_keys=True, separators=(',', ':')) + '\n')
    print(json.dumps({'flagged_records': result['flagged_corpus_record_count'], 'totals': result['totals'],
                      'exclusions': result['exclusions'], 'parse_errors': result['parse_errors']}, indent=2))


if __name__ == '__main__':
    main()
