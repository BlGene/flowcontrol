"""
This module check all lint scores, search for files from ../
"""
import os
from statistics import mean
from pylint.lint import Run


def lint():
    '''compute lint scores'''

    # lint along the watchtower
    py_files = []
    for root, _, files in os.walk("../"):
        for file in files:
            if file.endswith(".py"):
                py_files.append(os.path.join(root, file))

    lint_score = {}
    for filename in py_files:
        results = Run([filename,
                       '--disable=no-member,c-extension-no-member,import-error'
                       ],
                      do_exit=False)
        try:
            score = results.linter.stats['global_note']
        except (TypeError, KeyError):
            continue
        lint_score[filename] = score

    lint_score_sorted = dict(sorted(lint_score.items(), key=lambda item: item[1]))

    for filename, score in lint_score_sorted.items():
        print(filename.ljust(40), round(score, 2))

    print(f"\nmean score: {mean(lint_score.values())}")


if __name__ == "__main__":
    lint()
