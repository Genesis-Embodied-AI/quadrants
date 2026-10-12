"""Compare archived binaries or test workloads with matched installed dependencies.

The workflow runs old/new and new/old on separate VMs to expose order effects. This is a
source-version versus environment discriminator, not a bisect or proof that a specific PR is causal.
Workload mode holds the October binary fixed and compares the common tests with the full October suite.
Snapshots mode compares the complete matching tests and binary from each historical commit.
Dataclass-ablation mode removes only the added dataclass tests from PRs #842 and #867, keeping the October binary.
Expanded-binary mode retains unchanged test definitions from modified files, with the August test source fixed.
Pydantic mode holds the October binary and full tests fixed and exchanges only Pydantic and pydantic-core versions.
Process-batches mode holds the October binary/tests fixed and restarts pytest after each group of 12 test files.
Owner-guard mode protects ndarray deletion across runtime resets and removes the two forced collections per test.
"""
import argparse
import ast
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time


def command(args, **kwargs):
    print('+', shlex.join(str(a) for a in args), flush=True)
    subprocess.run([str(a) for a in args], check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('old', 'new', 'old-wheels', 'new-wheels', 'work'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--comparison',
                        choices=('binary', 'expanded-binary', 'workload', 'snapshots', 'gc-profile',
                                 'dataclass-ablation', 'pydantic', 'process-batches', 'owner-guard',
                                 'owner-guard-candidate', 'gc-freeze-candidate', 'process-batches-candidate',
                                 'production-batches-candidate', 'import-batch', 'import-batch-census', 'runtime-growth', 'kernel-registry', 'kernel-lifetime', 'gc-native', 'gc-walk', 'simt-only', 'python-gc', 'gc-order', 'gc-cpu', 'gc-memory', 'runtime-cycles', 'runtime-unique-cycles', 'runtime-allocation-cycles', 'runtime-materialize-cycles', 'runtime-standalone-cycles', 'runtime-standalone-exit-cycles', 'runtime-materialize-trace-cycles', 'runtime-no-mtlheap-cycles', 'runtime-no-argument-buffers-cycles', 'runtime-api-trace-cycles', 'runtime-no-root-cycles', 'runtime-no-root-trace-cycles'),
                        default='binary')
    parser.add_argument('--requirements', type=Path, help='Optional historically recorded dependency versions')
    parser.add_argument('--reference', type=Path, help='Completed full-suite artifact for candidate coverage checks')
    parser.add_argument('--order', required=True)
    args = parser.parse_args()
    cycle_comparison = args.comparison
    materialize_cycles = args.comparison in ('runtime-materialize-cycles', 'runtime-materialize-trace-cycles')
    allocation_cycles = args.comparison == 'runtime-allocation-cycles'
    unique_cycles = args.comparison == 'runtime-unique-cycles'
    runtime_cycles = args.comparison in ('runtime-cycles', 'runtime-unique-cycles', 'runtime-allocation-cycles', 'runtime-materialize-cycles', 'runtime-standalone-cycles', 'runtime-standalone-exit-cycles', 'runtime-materialize-trace-cycles', 'runtime-no-mtlheap-cycles', 'runtime-no-argument-buffers-cycles', 'runtime-api-trace-cycles', 'runtime-no-root-cycles', 'runtime-no-root-trace-cycles')
    cpu_gc = args.comparison == 'gc-cpu'
    order_gc = args.comparison == 'gc-order'
    memory_gc = args.comparison in ('gc-memory', 'runtime-cycles', 'runtime-unique-cycles', 'runtime-allocation-cycles', 'runtime-materialize-cycles', 'runtime-standalone-cycles', 'runtime-standalone-exit-cycles', 'runtime-materialize-trace-cycles', 'runtime-no-mtlheap-cycles', 'runtime-no-argument-buffers-cycles', 'runtime-api-trace-cycles', 'runtime-no-root-cycles', 'runtime-no-root-trace-cycles')
    walk_gc = args.comparison in ('gc-walk', 'gc-memory', 'runtime-cycles', 'runtime-unique-cycles', 'runtime-allocation-cycles', 'runtime-materialize-cycles', 'runtime-standalone-cycles', 'runtime-standalone-exit-cycles', 'runtime-materialize-trace-cycles', 'runtime-no-mtlheap-cycles', 'runtime-no-argument-buffers-cycles', 'runtime-api-trace-cycles', 'runtime-no-root-cycles', 'runtime-no-root-trace-cycles')
    python_gc = args.comparison == 'python-gc'
    simt_only = args.comparison == 'simt-only'
    native_gc = args.comparison in ('gc-native', 'gc-walk', 'simt-only', 'gc-order', 'gc-cpu', 'gc-memory')
    if native_gc or python_gc or runtime_cycles:
        args.comparison = 'runtime-growth'
    lifetime_only = args.comparison == 'kernel-lifetime'
    if lifetime_only:
        args.comparison = 'kernel-registry'
    production_batches = args.comparison == 'production-batches-candidate'
    candidate_only = args.comparison.endswith('-candidate')
    if candidate_only:
        if not args.reference or not args.requirements:
            parser.error('Candidate-only mode requires a completed reference artifact and historical dependency pins')
        args.comparison = 'process-batches' if production_batches else args.comparison.removesuffix('-candidate')
    if args.comparison == 'pydantic' and not args.requirements:
        parser.error('Pydantic comparison requires the historical October dependency pins.')
    expected = {'common', 'full'} if args.comparison == 'workload' else {'old', 'new'}
    if args.comparison == 'gc-profile':
        expected = {'new'}
    if args.comparison == 'dataclass-ablation':
        expected = {'pruned', 'full'}
    if args.comparison == 'process-batches':
        expected = {'batched'} if candidate_only else {'full', 'batched'}
    if args.comparison == 'owner-guard':
        expected = {'guarded'} if candidate_only else {'full', 'guarded'}
    if args.comparison == 'gc-freeze':
        expected = {'frozen'}
    if args.comparison.startswith('import-batch'):
        expected = {'small', 'all'}
    if args.comparison == 'runtime-growth':
        expected = {'cpu'} if cpu_gc else {'synthetic'} if python_gc else {'simt'} if simt_only else {'growth'}
    if args.comparison == 'kernel-registry':
        expected = {'strong', 'weak'}
    labels = args.order.split('-')
    if len(labels) != len(expected) or set(labels) != expected:
        parser.error(f'--order must contain each of {sorted(expected)} exactly once')
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
    expanded = None
    if args.comparison == 'expanded-binary':
        expanded = json.loads(Path(__file__).with_name('common_test_nodes.json').read_text())
        if {entry['path'] for entry in expanded['files']} != set(ignored):
            raise RuntimeError('Expanded test manifest does not cover exactly the changed test files.')
        for label, root in [('old', old), ('new', new)]:
            actual_ref = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
            if actual_ref != expanded[label + '_ref']:
                raise RuntimeError(f'{label}: source commit does not match the expanded test manifest.')
            for entry in expanded['files']:
                path = root / entry['path']
                expected_blob = entry[label + '_blob']
                if expected_blob is None:
                    if path.exists():
                        raise RuntimeError(f'Expected deleted file: {path}')
                    continue
                contents = path.read_bytes()
                actual_blob = hashlib.sha1(f'blob {len(contents)}\0'.encode() + contents).hexdigest()
                if actual_blob != expected_blob:
                    raise RuntimeError(f'Source content does not match expanded test manifest: {path}')
        (output / 'expanded_binary_selection.json').write_text(json.dumps(expanded, indent=2) + '\n')
    deselected = []
    if args.comparison == 'dataclass-ablation':
        mapping = json.loads(Path(__file__).with_name('dataclass_test_additions.json').read_text())
        added = {name for pr in mapping for name in pr['added_test_functions']}
        function_sets = []
        for root in (old, new):
            module = ast.parse((root / 'tests/python/test_py_dataclass.py').read_text())
            function_sets.append({node.name for node in module.body
                                  if isinstance(node, ast.FunctionDef) and node.name.startswith('test_')})
        old_functions, new_functions = function_sets
        if len(added) != 69 or new_functions - old_functions != added or old_functions - new_functions:
            raise RuntimeError('The proposed ablation does not exactly match the added dataclass test functions.')
        if any(old_name.startswith(name) for old_name in old_functions for name in added):
            raise RuntimeError('A deselection prefix would also exclude an older test.')
        deselected = [f'python/test_py_dataclass.py::{name}' for name in sorted(added)]
        (output / 'dataclass_ablation.json').write_text(json.dumps(mapping, indent=2) + '\n')

    # Resolve all dependencies once, then install that exact environment with only the wheel exchanged.
    pythons, wheels = {}, {}
    for label, folder in [('old', args.old_wheels), ('new', args.new_wheels)]:
        found = list(folder.resolve().glob('*.whl'))
        if len(found) != 1:
            raise RuntimeError(f'Expected one archived wheel in {folder}, found {len(found)}.')
        wheels[label] = found[0]
        env = work / ('venv-' + label)
        command(['uv', 'venv', '--python', '3.10', '--seed', env])
        pythons[label] = env / 'bin/python'
    if args.comparison == 'pydantic':
        wheels['old'] = wheels['new']
    install = ['uv', 'pip', 'install', '--python', pythons['old'], '--group', f'{old}/pyproject.toml:test']
    if args.requirements:
        install += ['-r', args.requirements.resolve()]
        (output / 'requested-dependencies.txt').write_text(args.requirements.read_text())
    command(install + [wheels['old']])
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
    if args.comparison == 'pydantic':
        versions = {'old': {'pydantic': '2.13.4', 'pydantic-core': '2.46.4'},
                    'new': {'pydantic': '2.13.5', 'pydantic-core': '2.46.5'}}
        command(['uv', 'pip', 'install', '--python', pythons['old'], '--no-deps',
                 *(f'{name}=={version}' for name, version in versions['old'].items())])
        baseline = dict(line.split('==', 1) for line in deps.read_text().splitlines() if line)
        for label, python in pythons.items():
            command(['uv', 'pip', 'check', '--python', python])
            frozen = subprocess.check_output(['uv', 'pip', 'freeze', '--python', str(python)], text=True)
            actual = dict(line.split('==', 1) for line in frozen.splitlines()
                          if line and not line.lower().startswith('quadrants'))
            if actual != {**baseline, **versions[label]}:
                raise RuntimeError(f'{label}: packages differ beyond the requested Pydantic pair.')
            (output / f'{label}-dependencies.txt').write_text(frozen)
        (output / 'pydantic_versions.json').write_text(json.dumps(versions, indent=2) + '\n')
    metadata = {label: dict(wheel=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                for label, path in wheels.items()}
    (output / 'wheels.json').write_text(json.dumps(metadata, indent=2) + '\n')
    if candidate_only or args.comparison == 'kernel-registry':
        if not args.reference:
            parser.error('Kernel registry comparison requires a full-suite reference')
        reference = args.reference.resolve()
        reference_wheels = json.loads((reference / 'wheels.json').read_text())
        reference_results = json.loads((reference / 'results.json').read_text())
        full = next(result for result in reference_results if result['arm'] == 'full')
        if (reference_wheels['new'] != metadata['new'] or
                (reference / 'dependencies.txt').read_bytes() != deps.read_bytes() or
                full['returncode'] != 0 or full['binary'] != 'new' or full['test_source'] != 'new'):
            raise RuntimeError('Reference is not a passing full October suite with the same wheel and dependencies')
        shutil.copy2(reference / 'full.xml', output / 'full.xml')
        (output / 'reference_result.json').write_text(json.dumps(full, indent=2) + '\n')

    results = []
    for label in labels:
        fixed_binary = args.comparison in (
            'workload', 'dataclass-ablation', 'pydantic', 'process-batches', 'owner-guard', 'gc-freeze',
            'import-batch', 'import-batch-census', 'runtime-growth', 'kernel-registry')
        binary = 'new' if fixed_binary else label
        test_root = (new if label == 'full' or
                     (args.comparison in ('snapshots', 'gc-profile') and label == 'new') else old)
        if args.comparison in ('dataclass-ablation', 'pydantic', 'process-batches', 'owner-guard', 'gc-freeze',
                               'import-batch', 'import-batch-census', 'runtime-growth', 'kernel-registry'):
            test_root = new
        python = pythons[label] if args.comparison == 'pydantic' else pythons[binary]
        env = dict(os.environ, PATH=f'{python.parent}:{os.environ["PATH"]}', VIRTUAL_ENV=str(python.parent.parent),
                   PYTHONHASHSEED='0', QD_TEST_THREADS='3', QD_FILE_TIMING='1',
                   QD_FILE_TIMING_OUTPUT=str(output / f'{label}-file-timing.md'))
        subset = args.comparison == 'binary' or label == 'common'
        options = [f'--ignore={rel}' for rel in ignored] if subset else []
        if expanded is not None:
            options += [f'--ignore={rel}' for rel in expanded['ignored_files']]
            options += [f'--deselect={nodeid}' for nodeid in expanded['deselected_nodes']]
        if args.comparison == 'dataclass-ablation' and label == 'pruned':
            options += [f'--deselect={nodeid}' for nodeid in deselected]
        options += [f'--basetemp={work / (label + "-pytest-tmp")}', f'--junitxml={output / (label + ".xml")}']
        if args.comparison in ('gc-profile', 'process-batches', 'owner-guard', 'gc-freeze',
                               'import-batch', 'import-batch-census', 'runtime-growth', 'kernel-registry'):
            options += ['-p', 'gc_profile']
            env['PYTHONPATH'] = str(Path(__file__).resolve().parent) + os.pathsep + env.get('PYTHONPATH', '')
            env['QD_GC_PROFILE_DIR'] = str(output if args.comparison == 'gc-profile' else output / (label + '-gc'))
        if args.comparison.startswith('import-batch'):
            from import_batch_control import prepare

            prepare(test_root, output, label, env, args.comparison.endswith('-census'))
            options += ['-p', 'import_batch']
        if args.comparison in ('runtime-growth', 'kernel-registry'):
            from import_batch_control import prepare_growth

            prepare_growth(test_root, output, env)
            if simt_only:
                selection = output / 'simt-selection.json'
                selection.write_text(json.dumps([str(test_root / 'tests/python/test_simt.py')]) + '\n')
                env['QD_BATCH_SELECTION'] = str(selection)
            options += ['-p', 'import_batch']
        if args.comparison == 'kernel-registry':
            env['QD_KERNEL_REGISTRY'] = label
            options += ['-p', 'kernel_registry_probe']
        if label == 'guarded':
            options += ['-p', 'ndarray_owner_guard']
        if label == 'frozen':
            options += ['-p', 'gc_freeze_baseline']
        if native_gc:
            env['QD_GC_NATIVE'] = '1'
        if order_gc:
            options += ['-p', 'gc_order_control']
        if memory_gc:
            env['QD_GC_MEMORY'] = '1'
            env['PYTHONFAULTHANDLER'] = '1'
        if walk_gc:
            env['QD_GC_WALK'] = '1'
            command([python, Path(__file__).with_name('gc_walk_probe.py').resolve()], env=env, cwd=test_root)
        env['PYTEST_ADDOPTS'] = shlex.join(options)
        if python_gc:
            from import_batch_control import run_bounded

            invocation = [str(python), str(Path(__file__).with_name('python_gc_synthetic.py').resolve()),
                          str(output / 'synthetic-gc')]
            start = time.monotonic()
            code = run_bounded(invocation, test_root, env, output, label)
            results.append(dict(arm=label, comparison='python-gc', seconds=time.monotonic() - start, returncode=code))
            (output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
            continue
        env['QD_LIB_DIR'] = subprocess.check_output(
            [str(python), '-c', 'import quadrants; print(quadrants.__path__[0] + "/_lib/runtime")'],
            text=True, env=env, cwd=test_root).splitlines()[-1]
        if runtime_cycles:
            if env.get('QD_CYCLE_RAW_DEVICE') == '1' and env.get('QD_RAW_DEVICE_WORK') == 'private-bda-runtime-no-fill':
                from native_crash_trace import build

                build(output / 'native-crash', env)
            if env.get('QD_CYCLE_RAW_DEVICE') == '1' and env.get('QD_RAW_DEVICE_WORK', '').startswith('metal-'):
                from native_metal_heap import build

                build(output / 'native-metal', env)
            if cycle_comparison in ('runtime-materialize-trace-cycles', 'runtime-api-trace-cycles', 'runtime-no-root-trace-cycles'):
                from vulkan_api_trace import install_trace

                install_trace(env['QD_LIB_DIR'], output / 'vulkan-api-trace', env)
            if cycle_comparison == 'runtime-no-mtlheap-cycles':
                env['MVK_CONFIG_USE_MTLHEAP'] = '0'
            elif cycle_comparison == 'runtime-no-argument-buffers-cycles':
                env['MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS'] = '0'
            env['QD_CYCLE_SKIP_EMPTY_ROOT'] = '1' if cycle_comparison in ('runtime-no-root-cycles', 'runtime-no-root-trace-cycles') else '0'
            env['QD_CYCLE_MATERIALIZE_ONLY'] = '1' if materialize_cycles else '0'
            env['QD_CYCLE_UNIQUE'] = '1' if unique_cycles else '0'
            env['QD_CYCLE_ALLOCATIONS'] = '1' if allocation_cycles else '0'
            from import_batch_control import run_bounded
            from memory_latency_watch import memory_watch

            exit_recovery = cycle_comparison == 'runtime-standalone-exit-cycles'
            env['QD_REPRO_EXIT_ON_SLOW'] = '1' if exit_recovery else '0'
            cycle_script = 'raw_runtime_repro.py' if cycle_comparison.startswith('runtime-standalone-') else 'runtime_gc_cycles.py'
            invocation = [str(python), str(Path(__file__).with_name(cycle_script).resolve()),
                          str(output / 'runtime-cycles')]
            start = time.monotonic()
            from crash_reports import crash_reports

            with crash_reports(output / 'crash-reports'), memory_watch(
                    python, output / 'memory-control', env, recovery_seconds=180 if exit_recovery else 0):
                code = run_bounded(invocation, test_root, env, output, label)
            results.append(dict(arm=label, comparison=cycle_comparison, seconds=time.monotonic() - start,
                                returncode=code))
            (output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
            continue
        invocation = [str(python), 'tests/run_tests.py', '-v', '-r', '1', '--arch', 'cpu' if cpu_gc else 'vulkan',
                      '-t', '3', '-m', 'not needs_torch']
        if production_batches:
            invocation = [str(python), str(Path(__file__).with_name('vulkan_batches.py').resolve()),
                          '--output', str(output / 'production-batches'), '--', *invocation[2:]]
        if args.comparison == 'kernel-registry':
            probe_start = time.monotonic()
            command([python, Path(__file__).with_name('kernel_registry_probe.py').resolve(),
                     output / f'{label}-lifetime.json'], env=env, cwd=test_root)
            if lifetime_only:
                results.append(dict(arm=label, comparison='kernel-lifetime', seconds=time.monotonic() - probe_start,
                                    returncode=0))
                (output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
                continue
        print(f'ARM {label}: binary={binary}, tests={test_root}; {shlex.join(invocation)}', flush=True)
        start = time.monotonic()
        with ExitStack() as stack:
            if memory_gc:
                from crash_reports import crash_reports

                stack.enter_context(crash_reports(output / 'crash-reports'))
            if memory_gc:
                from memory_latency_watch import memory_watch

                stack.enter_context(memory_watch(python, output / 'memory-control', env))
            if native_gc:
                from native_gc_watch import NativeGCWatch

                stack.enter_context(NativeGCWatch(output / (label + '-gc')))
            if label == 'guarded':
                from ndarray_owner_guard import without_forced_collections

                stack.enter_context(without_forced_collections(test_root, output))
            if label == 'batched' and not production_batches:
                from process_batches import run_batched

                code = run_batched(invocation, test_root, env, output, work)
            elif args.comparison.startswith('import-batch') or args.comparison in ('runtime-growth', 'kernel-registry'):
                from import_batch_control import run_bounded

                code = run_bounded(invocation, test_root, env, output, label)
            else:
                with (output / f'{label}.log').open('w') as log:
                    proc = subprocess.Popen(invocation, cwd=test_root, env=env, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, text=True, bufsize=1)
                    for line in proc.stdout:
                        print(line, end='', flush=True)
                        log.write(line)
                    code = proc.wait()
        if production_batches:
            shutil.copy2(output / 'production-batches/results.xml', output / 'batched.xml')
        results.append(dict(arm=label, comparison='gc-cpu' if cpu_gc else args.comparison, binary=binary,
                            test_source='new' if test_root == new else 'old',
                            seconds=time.monotonic() - start, returncode=code))
        (output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2), flush=True)
    if args.comparison == 'process-batches':
        from process_batches import verify_coverage

        verify_coverage(output)
    if args.comparison in ('owner-guard', 'gc-freeze'):
        from process_batches import verify_coverage

        verify_coverage(output, 'frozen' if args.comparison == 'gc-freeze' else 'guarded')
    if args.comparison.startswith('import-batch'):
        from import_batch_control import verify

        verify(output)
    if args.comparison == 'kernel-registry' and not lifetime_only:
        from process_batches import verify_coverage

        for result in results:
            if result['returncode'] == 0:
                verify_coverage(output, result['arm'])
    if any(r['returncode'] for r in results):
        raise SystemExit('An arm failed: inspect logs and test counts before interpreting durations.')


if __name__ == '__main__':
    main()
