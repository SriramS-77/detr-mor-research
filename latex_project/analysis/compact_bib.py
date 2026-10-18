"""Make references.bib render cleanly and compactly with sn-basic.bst.

* arXiv-only entries: ``journal = {arXiv preprint arXiv:XXXX.XXXXX}`` loses its
  dot under sn-basic's journal formatting, so the identifier moves to
  ``eprint`` / ``archivePrefix`` (rendered as arXiv:XXXX.XXXXX) and the
  redundant 10.48550 DOI is dropped.
* ``url`` fields are dropped: DOIs are kept wherever they exist, and the long
  OpenReview / CVF links mostly lengthen the reference list.

The verified source (written by the citation-verification pass) is kept as
references_verified.bib; this script regenerates references.bib from it.
"""

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 'references_verified.bib')
DST = os.path.join(HERE, '..', 'references.bib')


def main():
    text = open(SRC, encoding='utf-8').read()
    entries = re.split(r'\n(?=@)', text.strip())
    out = []
    for entry in entries:
        arxiv = re.search(r'journal\s*=\s*\{arXiv preprint arXiv:([0-9.]+)\}', entry)
        if arxiv:
            entry = re.sub(r'journal\s*=\s*\{arXiv preprint arXiv:[0-9.]+\}',
                           'journal = {arXiv preprint},\n  eprint  = {%s},\n'
                           '  archivePrefix = {arXiv}' % arxiv.group(1), entry)
            entry = re.sub(r',?\n\s*doi\s*=\s*\{10\.48550/[^}]*\}', '', entry)
        entry = re.sub(r',?\n\s*url\s*=\s*\{[^}]*\}', '', entry)
        out.append(entry)
    with open(DST, 'w', encoding='utf-8') as handle:
        handle.write('\n\n'.join(out) + '\n')
    print('{} entries -> {}'.format(len(out), DST))


if __name__ == '__main__':
    main()
