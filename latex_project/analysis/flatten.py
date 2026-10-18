"""Assemble latex_project/submission/: a single-file manuscript plus support files.

Springer Nature asks for one .tex source, so generated/numbers.tex (all values
and plot tables) is inlined in place of its \\input. The bibliography stays a
.bib file, which Editorial Manager compiles with sn-basic.bst.

Example::

    python latex_project/analysis/flatten.py
"""

import os
import shutil

PAPER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SUB = os.path.join(PAPER, 'submission')


def main():
    os.makedirs(SUB, exist_ok=True)
    tex = open(os.path.join(PAPER, 'main.tex'), encoding='utf-8').read()
    numbers = open(os.path.join(PAPER, 'generated', 'numbers.tex'), encoding='utf-8').read()
    marker = '\\input{generated/numbers}'
    assert tex.count(marker) == 1
    tex = tex.replace(marker, '%% ---- inlined from generated/numbers.tex ----\n'
                      + numbers.rstrip('\n') + '\n%% ---- end of inlined values ----')
    with open(os.path.join(SUB, 'manuscript.tex'), 'w', encoding='utf-8') as handle:
        handle.write(tex)
    for name in ('references.bib', 'sn-jnl.cls', 'sn-basic.bst', 'title_page.tex'):
        shutil.copy(os.path.join(PAPER, name), SUB)
    print('wrote', SUB)


if __name__ == '__main__':
    main()
