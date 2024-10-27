from collections import defaultdict

# This evaluation script is taken from https://github.com/psunlpgroup/XSemPLR/blob/main/model/UniPSP/metrics/exact_str_match/evaluator.py


def compute_exact_match(pred, gold):

    ## Aligment 3: replace ~ (Schema2QA only)
    pred = pred.replace(' = ', ' =~ ')
    pred = pred.replace('" org', '" ^^org')

    ## Alignment 2: replace ' '
    pred = pred.replace(' ', '')
    gold = gold.replace(' ', '')

    return pred == gold

# rename from EvaluateTool to MSchema2QAEvaluateTool

class MSchema2QAEvaluateTool(object):
    def __init__(self, args):
        self.args = args

    def evaluate(self, preds, golds):
        eval_dict = defaultdict(float)
        for pred, gold in zip(preds, golds):
            eval_dict["exact_match"] += compute_exact_match(pred, gold)
        for key in eval_dict:
            eval_dict[key] = eval_dict[key] / len(golds) if len(golds) else 0
        return eval_dict
