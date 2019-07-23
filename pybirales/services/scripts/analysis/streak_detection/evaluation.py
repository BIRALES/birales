import numpy as np
from skimage.measure import compare_ssim as ssim
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, \
    jaccard_similarity_score, mean_squared_error
import time


def evaluate_filter(truth_img, test_img, positives, exec_time, snr, thickness):
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
    score = 2 * (recall * reduction) / (recall + reduction)

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


def evaluate_detector(truth_img, test_img, candidates, exec_time, snr, thickness):
    # truth = truth_img.ravel()
    # truth[truth > 0.] = True
    # truth[truth <= 0.] = False

    truth = truth_img.ravel().astype('bool')

    prediction = np.zeros(shape=test_img.shape)

    if candidates:
        for c in candidates:
            prediction[c['channel'].astype(int), c['sample'].astype(int)] = True
    # positives = pos[:,0:1]
    # prediction[pos[:, :2].astype(int)] = True
    prediction = prediction.ravel()

    recall = recall_score(truth, prediction)
    reduction = (1 - (np.sum(prediction) / np.prod(truth_img.shape)))

    tn, fp, fn, tp = confusion_matrix(truth, prediction).ravel().astype('float64')
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    # harmonic mean of the recall and reduction rate
    score = 2 * (recall * reduction) / (recall + reduction)

    return {
        'jaccard': jaccard_similarity_score(truth, prediction),
        'f1': f1_score(truth, prediction),
        # 'precision': precision_score(truth, prediction, average='binary'),
        'precision': tp / (tp + fp),
        'recall': recall,
        'accuracy': accuracy_score(truth, prediction),
        'mse': mean_squared_error(truth, prediction),
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
