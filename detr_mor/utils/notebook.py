"""Load class / function definitions out of the RT-DETR notebooks.

The RT-DETR models live only in ``rt_detr.ipynb`` / ``rt_detr_mor.ipynb``.
Scripts that need them (latency profiling, routing diagnostics) import the
definitions from there, so they always run exactly the code that was trained.
"""

import ast
import json

_IMPORT_NODES = (ast.Import, ast.ImportFrom)
_DEF_NODES = (ast.FunctionDef, ast.ClassDef)


def _is_literal_assign(node):
    r"""``NAME = <literal>`` - module constants, never calls like torch.load."""
    if not isinstance(node, ast.Assign):
        return False
    # Plain names only: `x[i] = True` has a literal value but mutates state
    # built by statements that are deliberately not executed.
    if not all(isinstance(target, ast.Name) for target in node.targets):
        return False
    try:
        ast.literal_eval(node.value)
    except ValueError:
        return False
    return True


def load_notebook_namespace(notebook_path, stop_after='DETR', preset=None,
                            required=('DETR',)):
    r"""
    Execute the definition-only parts of a notebook and return its namespace.

    Code cells are walked in order. A cell is used only if it is import-only or
    defines a function / class, and even then only its imports, defs, classes,
    try-blocks (the optional mmcv import) and literal assignments run. Prints,
    checkpoint loading, smoke tests, training and evaluation calls are never
    executed.

    :param stop_after: stop once this name is defined; None walks every cell
    :param preset: names to seed the namespace with, for non-literal module
        constants a later definition refers to (e.g. a default argument)
    :param required: names that must exist afterwards
    :return: dict namespace
    """
    with open(notebook_path, encoding='utf-8') as handle:
        cells = json.load(handle)['cells']

    namespace = {'__name__': 'notebook_defs'}
    namespace.update(preset or {})
    for cell in cells:
        if cell['cell_type'] != 'code':
            continue
        tree = ast.parse(''.join(cell['source']))
        import_only = bool(tree.body) and all(
            isinstance(node, _IMPORT_NODES) for node in tree.body)
        defines = any(isinstance(node, _DEF_NODES) for node in tree.body)
        if not (import_only or defines):
            continue
        kept = [node for node in tree.body
                if isinstance(node, _IMPORT_NODES + _DEF_NODES + (ast.Try,))
                or _is_literal_assign(node)]
        for node in kept:
            module = ast.Module(body=[node], type_ignores=[])
            try:
                exec(compile(module, notebook_path, 'exec'), namespace)
            except NameError:
                # A def whose default argument / decorator refers to a
                # non-literal constant that was deliberately not executed -
                # only ever the case in smoke-test / training cells. Anything
                # actually needed is caught by the `required` check below.
                if not isinstance(node, _DEF_NODES):
                    raise
        if stop_after is not None and stop_after in namespace:
            break

    missing = [name for name in required if name not in namespace]
    if missing:
        raise RuntimeError('{} did not define: {}'.format(
            notebook_path, ', '.join(missing)))
    return namespace
