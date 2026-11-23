"""Save each panel of a multi-panel figure as its own PNG (paper layout, 2026-11-23).

Call after the combined figure has been saved. Each panel is drawn alone (other panels and figure-level
footer text hidden), with its own tick labels switched on, and its "A. " title prefix dropped, so panels can
be placed and lettered freely. Output: <stem>_a.png, <stem>_b.png, ... next to the combined figure. Colour keys
that lived in a figure-level footer are not repeated; put them in the caption.
"""

import re


def save_panels(fig, out_dir, stem, dpi=150):
    axes = [a for a in fig.axes if a.get_visible()]
    if len(axes) < 2:
        return []
    for t in fig.texts:
        t.set_visible(False)
    saved = []
    for i, ax in enumerate(axes):
        for a in axes:
            a.set_visible(a is ax)
        title = ax.get_title()
        m = re.match(r"^([A-Z])\.\s+", title)
        letter = (m.group(1) if m else chr(65 + i)).lower()
        if m:
            ax.set_title(title[m.end():])
        ax.tick_params(labelleft=True, labelbottom=True)
        path = out_dir / f"{stem}_{letter}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        ax.set_title(title)
        saved.append(path)
    for a in axes:
        a.set_visible(True)
    return saved
