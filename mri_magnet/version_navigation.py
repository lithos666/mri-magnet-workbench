"""Share the small version selector with an immutable V5 installation.

The canonical component lives once in workbench.html. The V5 launcher injects
only this UI component; it never imports V6 physics into the V5 process.
"""
from pathlib import Path
import re


def _block(page, name):
    start='<!-- MRI_VERSION_'+name+'_START -->'
    end='<!-- MRI_VERSION_'+name+'_END -->'
    first=page.index(start);last=page.index(end,first)+len(end)
    return page[first:last]


def inject_version_navigation(page, canonical_html, version='v5'):
    if version not in ('v5','v6'):raise ValueError('Unsupported workbench version')
    if 'id="version-navigation"' in page:return page
    source=Path(canonical_html).read_text(encoding='utf-8')
    style=_block(source,'STYLE');markup=_block(source,'MARKUP');script=_block(source,'SCRIPT')
    markup=markup.replace('data-version="v6"','data-version="'+version+'"')
    if version=='v5':markup=markup.replace('V6 · 阵列互耦','V5 · 固定源')
    page,count=re.subn(r'<aside\b[^>]*>',lambda match:match[0]+markup,page,count=1)
    if not count:raise ValueError('V5 page has no navigation sidebar')
    page,count=re.subn(r'</head>',lambda match:style+match[0],page,count=1,flags=re.I)
    if not count:raise ValueError('V5 page has no HTML head')
    page,count=re.subn(r'</body>',lambda match:script+match[0],page,count=1,flags=re.I)
    if not count:raise ValueError('V5 page has no HTML body')
    return page
