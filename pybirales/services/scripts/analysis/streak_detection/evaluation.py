import time

import numpy as np
import pandas as pd
from skimage.measure import compare_ssim as ssim
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, \
    mean_squared_error


def new_e(truth_img, positives):
    truth = np.where(truth_img > 0, 1, 0)

    pred = np.zeros(shape=truth_img.shape)
    pred[positives] = True

    s = truth + pred
    b = truth - pred

    tp = len(s[s == 2])
    tn = len(s[s == 0])

    fp = len(b[b == -1])
    fn = len(b[b == 1])

    return float(tn), float(fp), float(fn), float(tp)


def evaluate_filter(truth_img, test_img, positives, exec_time, snr, thickness, name):
    """

    :param truth:
    :param prediction:
    :param positives: Location where the filter thinks there should be a target
    :return:
    """

    tn, fp, fn, tp = new_e(truth_img, positives)

    p = tp + fn
    n = tn + fp

    specificity = tn / n  # aka: specificity, selectivity or true negative rate
    sensitivity = tp / p  # aka: sensitivity, recall, hit rate, or true positive rate

    fpr = 1 - specificity  # aka: fall-out or false positive rate (FPR)
    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        precision = 0

    accuracy = (tp + tn) / (p + n)
    f1 = 2 * tp / (2 * tp + fp + fn)
    reduction = 1 - (fp + tp) / truth_img.size

    score = 2 * (sensitivity * specificity) / (sensitivity + specificity)

    return {
        'name': name,
        'f1': f1,
        'precision': precision,
        'accuracy': accuracy,
        'fpr': fpr,
        'tpr': sensitivity,
        'dt': exec_time,
        'nchans': truth_img.shape[0],
        'nsamples': truth_img.shape[1],
        'snr': snr,
        'dx': thickness,
        'recall': sensitivity,
        'specificity': specificity,
        'reduction': reduction * 100.,
        'score': score
    }


def evaluate_filter_old(truth_img, test_img, positives, exec_time, snr, thickness):
    """

    :param truth:
    :param prediction:
    :param positives: Location where the filter thinks there should be a target
    :return:
    """
    t1 = time.time()
    truth = truth_img.ravel().astype('bool')
    # truth[truth > 0.] = True
    # truth[truth <= 0.] = False

    # test_img[:] = False
    # test_img[positives] = True
    # prediction = test_img.ravel()

    prediction = np.zeros(shape=test_img.shape)
    # prediction[:] = False
    prediction[positives] = True
    prediction = prediction.ravel()

    ssim_score = ssim((truth).astype('float64'), (prediction).astype('float64'))

    recall = recall_score(truth, prediction)
    reduction = (1 - (np.sum(prediction) / np.prod(truth_img.shape)))
    tn, fp, fn, tp = confusion_matrix(truth, prediction).ravel().astype('float64')

    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    # harmonic mean of the recall and reduction rate
    score = 2 * (recall * specificity) / (recall + specificity)

    return {
        'jaccard': jaccard_similarity_score(truth, prediction),
        'f1': f1_score(truth, prediction),
        'precision': precision_score(truth, prediction, average='binary'),
        'recall': recall,
        'accuracy': accuracy_score(truth, prediction),
        'mse': mean_squared_error(truth, prediction),
        'ssim': ssim_score,
        'dt': exec_time,
        'nchans': truth_img.shape[0],
        'nsamples': truth_img.shape[1],
        'snr': snr,
        'dx': thickness,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'n_px': np.sum(prediction).astype(np.int),
        'reduction': reduction * 100.,
        'score': score
    }

def evaluate_detector(truth_img, test_img, candidates, exec_time, snr, thickness, name):
    # # truth = truth_img.ravel()
    # # truth[truth > 0.] = True
    # # truth[truth <= 0.] = False
    #
    # truth = truth_img.ravel().astype('bool')

    prediction = np.zeros(shape=test_img.shape).astype('bool')

    if candidates:
        for c in candidates:
            if isinstance(c, pd.DataFrame):
                prediction[c['channel'].astype(int), c['sample'].astype(int)] = True
            else:
                prediction[c[:, 0].astype(int), c[:, 1].astype(int)] = True
    # # positives = pos[:,0:1]
    # # prediction[pos[:, :2].astype(int)] = True
    # prediction = prediction.ravel()
    #
    # # recall = recall_score(truth, prediction)
    # reduction = (1 - (np.sum(prediction) / np.prod(truth_img.shape)))

    tn, fp, fn, tp = new_e(truth_img, prediction)

    p = tp + fn
    n = tn + fp

    specificity = tn / n  # aka: specificity, selectivity or true negative rate
    sensitivity = tp / p  # aka: sensitivity, recall, hit rate, or true positive rate

    fpr = 1 - specificity  # aka: fall-out or false positive rate (FPR)
    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        precision = 0

    accuracy = (tp + tn) / (p + n)
    f1 = 2 * tp / (2 * tp + fp + fn)
    reduction = 1 - (fp + tp) / truth_img.size

    score = 2 * (sensitivity * specificity) / (sensitivity + specificity)

    return {
        'name': name,
        'f1': f1,
        'precision': precision,
        'accuracy': accuracy,
        'fpr': fpr,
        'tpr': sensitivity,
        'dt': exec_time,
        'nchans': truth_img.shape[0],
        'nsamples': truth_img.shape[1],
        'snr': snr,
        'dx': thickness,
        'recall': sensitivity,
        'specificity': specificity,
        'reduction': reduction * 100.,
        'score': score
    }


def evaluate_detector_old(truth_img, test_img, candidates, exec_time, snr, thickness):
    # truth = truth_img.ravel()
    # truth[truth > 0.] = True
    # truth[truth <= 0.] = False

    truth = truth_img.ravel().astype('bool')

    prediction = np.zeros(shape=test_img.shape)

    if candidates:
        for c in candidates:
            if isinstance(c, pd.DataFrame):
                prediction[c['channel'].astype(int), c['sample'].astype(int)] = True
            else:
                prediction[c[:, 0].astype(int), c[:, 1].astype(int)] = True
    # positives = pos[:,0:1]
    # prediction[pos[:, :2].astype(int)] = True
    prediction = prediction.ravel()

    # recall = recall_score(truth, prediction)
    reduction = (1 - (np.sum(prediction) / np.prod(truth_img.shape)))

    tn, fp, fn, tp = confusion_matrix(truth, prediction).ravel().astype('float64')
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)  # aka recall

    # harmonic mean of the recall and reduction rate
    # score = 2 * (recall * reduction) / (recall + reduction)

    return {
        # 'jaccard': jaccard_similarity_score(truth, prediction),
        'f1': f1_score(truth, prediction),
        # 'precision': precision_score(truth, prediction, average='binary'),
        'precision': tp / (tp + fp),
        'recall': sensitivity,
        # 'accuracy': accuracy_score(truth, prediction),
        # 'mse': mean_squared_error(truth, prediction),
        'dt': exec_time,
        'nchans': truth_img.shape[0],
        'nsamples': truth_img.shape[1],
        'snr': snr,
        'dx': thickness,
        'sensitivity': sensitivity,
        'specificity': specificity,
        # 'n_px': np.sum(prediction).astype(np.int),
        'reduction': reduction * 100.,
        # 'score': score,
        'N': len(candidates)
    }
