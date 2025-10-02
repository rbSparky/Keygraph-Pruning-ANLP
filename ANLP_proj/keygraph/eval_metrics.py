from rouge_score import rouge_scorer
import string
import re


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b',' ',text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude =set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction,ground_truth):
    """Compute exact match between prediction and ground truth."""
    return 1.0 if normalize_answer(prediction)==normalize_answer(ground_truth)else 0.0


def compute_f1(prediction,ground_truth):
    """Compute F1 score between prediction and ground truth."""
    pred_tokens =normalize_answer(prediction).split()
    gt_tokens =normalize_answer(ground_truth).split()

    common_tokens =set(pred_tokens)&set(gt_tokens)

    if len(common_tokens)==0:
        return 0.0

    precision =len(common_tokens)/len(pred_tokens)if pred_tokens else 0.0
    recall =len(common_tokens)/len(gt_tokens)if gt_tokens else 0.0

    f1 =2 *precision *recall /(precision +recall)if(precision +recall)>0 else 0.0
    return f1


def compute_rouge_scores(prediction,ground_truth):
    """Compute ROUGE scores between prediction and ground truth."""
    scorer =rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'],use_stemmer =True)
    scores =scorer.score(ground_truth,prediction)

    return {
    'rouge1':scores['rouge1'].fmeasure,
    'rouge2':scores['rouge2'].fmeasure,
    'rougeL':scores['rougeL'].fmeasure}


def evaluate_prediction(task_type,prediction,ground_truths):
    """Evaluate a prediction based on the task type."""
    if task_type =="summarization":

        if isinstance(ground_truths,list):
            ground_truth =ground_truths[0]if ground_truths else ""
        else:
            ground_truth =ground_truths

        rouge_scores =compute_rouge_scores(prediction,ground_truth)
        return rouge_scores

    elif task_type in["qa","qasper"]:

        if not isinstance(ground_truths,list):
            ground_truths =[ground_truths]


        em_scores =[compute_exact_match(prediction,gt)for gt in ground_truths]
        f1_scores =[compute_f1(prediction,gt)for gt in ground_truths]


        max_em =max(em_scores)if em_scores else 0.0
        max_f1 =max(f1_scores)if f1_scores else 0.0

        return {
        'exact_match':max_em,
        'f1':max_f1}

    else:
        raise ValueError(f"Unsupported task type: {task_type}")