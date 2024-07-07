import os
from hiq import ensure_folder

def save_figure(fig, imgnum, filepath):
    script_name = os.path.splitext(os.path.basename(filepath))[0]
    filename = f"imgs/{script_name}_{imgnum}.png"
    ensure_folder(filename)
    fig.savefig(filename)
    print(f"Figure saved as {filename}")