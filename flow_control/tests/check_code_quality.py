import os
import glob
from operator import itemgetter
from pylint.lint import Run


def lint_sort(lint_scores):
    '''pretty print sorted lint scores'''
    num_files = 100
    res = dict(sorted(lint_scores.items(), key=itemgetter(1))[:num_files])
    for filename, score in res.items():
        print(filename.ljust(40), round(score, 2))


def lint():
    '''compute lint scores'''
    lint_score = {}
    files = sorted(glob.glob("../*/*.py"))
    for filename in files:
        results = Run([filename,
                       '--disable=no-member,c-extension-no-member,import-error'
                       ],
                      do_exit=False)
        try:
            score = results.linter.stats['global_note']
        except TypeError:
            continue
        lint_score[filename] = score

    for filename, score in lint_score.items():
        print(filename.ljust(40), score)
    return lint_score


if __name__ == "__main__":
    lint_scores = lint()
