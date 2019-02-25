from typing import List

from matplotlib.pyplot import subplots, subplots_adjust


def visualize_image_grid(images: List, captions: List, n_rows: int):
    """Show some images along with corresponding questions and answers"""
    n_cols = len(images) // n_rows
    fig, ax = subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    subplots_adjust(wspace=0.5)
    ax = ax.flatten()
    for i, ax_ in enumerate(ax):
        ax_.imshow(images[i])
        q = captions[i]
        ax_.set_title(f"{q['question'][:40]} \n {q['answer']}")
