"""Run all test files in successive fresh pytest processes; retain all logs and compare complete case outcomes.

Each invocation uses the original runner and fixtures. Selection happens before module import. Both ordinary
cleanup and interpreter shutdown run between groups; no synthetic successful reports or forced worker exits are used.
"""
from collections import Counter
import json
from pathlib import Path
import shlex
import subprocess
import time
import xml.etree.ElementTree as ET


def run_batched(invocation, test_root, env, output, work):
    root = test_root / 'tests/python'
    files = sorted({*root.rglob('test_*.py'), *root.rglob('*_test.py'), *root.rglob('*.ipynb')})
    if not files:
        raise RuntimeError('Empty batch file manifest')
    groups = [files[start:start + 12] for start in range(0, len(files), 12)]
    (output / 'batch_manifest.json').write_text(json.dumps(
        [[str(path.relative_to(test_root)) for path in group] for group in groups], indent=2) + '\n')
    folder = output / 'batches'
    folder.mkdir()
    records = []
    combined = ET.Element('testsuites')
    with (output / 'batched.log').open('w') as log:
        for index, group in enumerate(groups):
            label = f'batch-{index:02d}'
            selection = folder / (label + '-selection.json')
            selection.write_text(json.dumps([str(path.resolve()) for path in group]) + '\n')
            xml = folder / (label + '.xml')
            options = [arg for arg in shlex.split(env['PYTEST_ADDOPTS'])
                       if not arg.startswith(('--basetemp=', '--junitxml='))]
            options += ['-p', 'batch_selection', f'--basetemp={work / (label + "-tmp")}', f'--junitxml={xml}']
            batch_env = dict(env, PYTEST_ADDOPTS=shlex.join(options), QD_BATCH_SELECTION=str(selection),
                             QD_GC_PROFILE_DIR=str(folder / label))
            message = f'BATCH {index + 1}/{len(groups)}: {[str(path.relative_to(root)) for path in group]}\n'
            print(message, end='', flush=True)
            log.write(message)
            start = time.monotonic()
            with (folder / (label + '.log')).open('w') as batch_log:
                proc = subprocess.Popen(invocation, cwd=test_root, env=batch_env, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in proc.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    batch_log.write(line)
                code = proc.wait()
            record = dict(batch=index, seconds=time.monotonic() - start, returncode=code, xml_exists=xml.exists())
            if xml.exists():
                suites = ET.parse(xml).getroot()
                if suites.tag == 'testsuite':
                    combined.append(suites)
                else:
                    combined.extend(suites)
            records.append(record)
            print('BATCH RESULT', json.dumps(record), flush=True)
            (output / 'batch_results.json').write_text(json.dumps(records, indent=2) + '\n')
    ET.ElementTree(combined).write(output / 'batched.xml', encoding='utf-8', xml_declaration=True)
    return int(any(record['returncode'] or not record['xml_exists'] for record in records))


def verify_coverage(output):
    def cases(label):
        counts = Counter()
        for case in ET.parse(output / (label + '.xml')).iter('testcase'):
            outcome = next((name for name in ('error', 'failure', 'skipped') if case.find(name) is not None), 'passed')
            counts[(case.get('classname'), case.get('name'), outcome)] += 1
        return counts

    full, batched = cases('full'), cases('batched')
    differences = [dict(case=key, full=full[key], batched=batched[key])
                   for key in sorted(full.keys() | batched.keys()) if full[key] != batched[key]]
    (output / 'batch_coverage.json').write_text(json.dumps(dict(
        full_cases=sum(full.values()), batched_cases=sum(batched.values()), differences=differences), indent=2) + '\n')
    if not full or differences:
        raise RuntimeError('Batched/full outcomes differ; inspect batch_coverage.json before interpreting timings.')
