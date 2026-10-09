"""Compare archived binaries on the same unchanged test files and installed dependencies.

The workflow runs old/new and new/old on separate VMs to expose order effects. This is a
source-version versus environment discriminator, not a bisect or proof that a specific PR is causal.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time


def command(args, **kwargs):
    print('+', shlex.join(str(a) for a in args), flush=True)
    subprocess.run([str(a) for a in args], check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('old', 'new', 'old-wheels', 'new-wheels', 'work'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--order', choices=('old-new', 'new-old'), required=True)
    args = parser.parse_args()
    old, new, work = args.old.resolve(), args.new.resolve(), args.work.resolve()
    output = work / 'output'
    output.mkdir(parents=True, exist_ok=True)
    temp = work / 'tmp'
    temp.mkdir(exist_ok=True)
    os.environ['TMPDIR'] = str(temp)
    command(['sw_vers'])
    command(['sysctl', 'hw.ncpu', 'hw.memsize'])

    # Test discovery and fixture behavior must be identical, independently of the wheel under test.
    helpers = ['tests/run_tests.py', 'tests/test_utils.py', 'tests/pytest.ini']
    helpers += [str(p.relative_to(old)) for p in (old / 'tests').rglob('conftest.py')]
    for rel in helpers:
        if not (new / rel).exists() or (old / rel).read_bytes() != (new / rel).read_bytes():
            raise RuntimeError(f'Test infrastructure differs: {rel}; revise the experiment before running it.')
    common, ignored = [], []
    for path in sorted((old / 'tests/python').rglob('test_*.py')):
        rel = path.relative_to(old)
        if (new / rel).exists() and path.read_bytes() == (new / rel).read_bytes():
            common.append(str(rel))
        else:
            ignored.append(str(rel))
    if 'tests/python/test_simt.py' not in common:
        raise RuntimeError('SIMT tests are not identical between source snapshots.')
    (output / 'test_manifest.json').write_text(json.dumps(dict(common=common, ignored=ignored), indent=2) + '\n')

    # Resolve all dependencies once, then install that exact environment with only the wheel exchanged.
    pythons, wheels = {}, {}
    for label, folder in [('old', args.old_wheels), ('new', args.new_wheels)]:
        found = list(folder.resolve().glob('*.whl'))
        if len(found) != 1:
            raise RuntimeError(f'Expected one archived wheel in {folder}, found {len(found)}.')
        wheels[label] = found[0]
        env = work / ('venv-' + label)
        command(['uv', 'venv', '--python', '3.10', env])
        pythons[label] = env / 'bin/python'
    command(['uv', 'pip', 'install', '--python', pythons['old'],
             '--group', f'{old}/pyproject.toml:test', wheels['old']])
    frozen = subprocess.check_output(['uv', 'pip', 'freeze', '--python', str(pythons['old'])], text=True)
    deps = output / 'dependencies.txt'
    deps.write_text('\n'.join(line for line in frozen.splitlines()
                             if not line.lower().startswith('quadrants')) + '\n')
    command(['uv', 'pip', 'install', '--python', pythons['new'], '-r', deps, wheels['new']])
    for label, python in pythons.items():
        frozen = subprocess.check_output(['uv', 'pip', 'freeze', '--python', str(python)], text=True)
        actual = '\n'.join(line for line in frozen.splitlines() if not line.lower().startswith('quadrants')) + '\n'
        if actual != deps.read_text():
            raise RuntimeError(f'{label}: dependency versions differ; comparison is invalid.')
    metadata = {label: dict(wheel=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                for label, path in wheels.items()}
    (output / 'wheels.json').write_text(json.dumps(metadata, indent=2) + '\n')

    results = []
    for label in args.order.split('-'):
        python = pythons[label]
        env = dict(os.environ, PATH=f'{python.parent}:{os.environ["PATH"]}', VIRTUAL_ENV=str(python.parent.parent),
                   PYTHONHASHSEED='0', QD_TEST_THREADS='3', QD_FILE_TIMING='1',
                   QD_FILE_TIMING_OUTPUT=str(output / f'{label}-file-timing.md'))
        options = [f'--ignore={rel}' for rel in ignored]
        options += [f'--basetemp={work / (label + "-pytest-tmp")}', f'--junitxml={output / (label + ".xml")}']
        env['PYTEST_ADDOPTS'] = shlex.join(options)
        env['QD_LIB_DIR'] = subprocess.check_output(
            [str(python), '-c', 'import quadrants; print(quadrants.__path__[0] + "/_lib/runtime")'],
            text=True, env=env, cwd=old).splitlines()[-1]
        invocation = [str(python), 'tests/run_tests.py', '-v', '-r', '1', '--arch', 'vulkan',
                      '-t', '3', '-m', 'not needs_torch']
        print(f'ARM {label}: {shlex.join(invocation)}', flush=True)
        start = time.monotonic()
        with (output / f'{label}.log').open('w') as log:
            proc = subprocess.Popen(invocation, cwd=old, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in proc.stdout:
                print(line, end='', flush=True)
                log.write(line)
            code = proc.wait()
        results.append(dict(arm=label, seconds=time.monotonic() - start, returncode=code))
        (output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2), flush=True)
    if any(r['returncode'] for r in results):
        raise SystemExit('An arm failed: inspect logs and test counts before interpreting durations.')


if __name__ == '__main__':
    main()
