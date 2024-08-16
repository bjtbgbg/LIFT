import numpy as np
from sklearn import metrics
import json
import os

def AUC(output, target):
    assert output.shape[0] == target.shape[0]
    class_auc = []
    for i in range(output.shape[1]):
        # y_true = target[:, i].flatten()
        y_true = (target == i)
        y_pred = output[:, i]
        # if np.max(y_true) > 0:
        class_auc.append(metrics.roc_auc_score(y_true, y_pred))
        # else:
        #     class_auc.append(0)
        # class_auc.append(metrics.roc_auc_score(y_true, y_pred))
    return np.array(class_auc)

def ACC(output, target):
    y_pred = output.argmax(1)
    y_true = target.flatten()
    y_pred = y_pred.flatten()
    return metrics.accuracy_score(y_true, y_pred)

def Cohen_Kappa(output, target):
    y_pred = output.argmax(1)
    y_true = target.flatten()
    y_pred = y_pred.flatten()
    return metrics.cohen_kappa_score(y_true, y_pred)

def F1_score(output, target):
    y_pred = output.argmax(1)
    y_true = target.flatten()
    y_pred = y_pred.flatten()
    return metrics.f1_score(y_true, y_pred, average='macro')

def Recall(output, target):
    y_pred = output.argmax(1)
    y_true = target.flatten()
    y_pred = y_pred.flatten()
    return metrics.recall_score(y_true, y_pred, average='macro')

def Specificity(output, target):
    con_mat = confusion_matrix(output, target)
    spe = []
    n = con_mat.shape[0]
    for i in range(n):
        number = np.sum(con_mat[:,:])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i,:]) - tp
        fp = np.sum(con_mat[:,i]) - tp
        tn = number - tp - fn - fp
        spe1 = tn / (tn + fp)
        spe.append(spe1)
    return np.array(spe)

def Precision(output, target):
    y_pred = output.argmax(1)
    y_true = target.flatten()
    y_pred = y_pred.flatten()
    return metrics.precision_score(y_true, y_pred, average='macro')


def Npv_cls(output, target):
    con_mat = confusion_matrix(output, target)
    npv = []
    n = con_mat.shape[0]
    for i in range(n):
        number = np.sum(con_mat[:,:])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i,:]) - tp
        fp = np.sum(con_mat[:,i]) - tp
        tn = number - tp - fn - fp
        npv1 = tn / (tn + fn)
        npv.append(npv1)
    return np.array(npv)

def cls_report(output, target, output_dict=False):
    y_pred = output.argmax(1)
    y_true = target.flatten()
    y_pred = y_pred.flatten()
    return metrics.classification_report(y_true, y_pred, output_dict=output_dict, digits=4)


def confusion_matrix(output, target):
    y_pred = output.argmax(1)
    y_true = target.flatten()
    y_pred = y_pred.flatten()
    return metrics.confusion_matrix(y_true, y_pred)

def write_score2json(score_info, val_anno_file, results_dir):
    score_info = score_info.astype(np.float)
    score_list = []
    anno_info = np.loadtxt(val_anno_file, dtype=np.str_)
    for idx, item in enumerate(anno_info):
        id = item[0].rsplit('/', 1)[-1]
        label = int(item[1])
        score = list(score_info[idx])
        pred = score.index(max(score))
        pred_info = {
            'image_id': id,
            'label': label,
            'prediction': pred,
            # 'benign/maglinant': int(pred in [1,3,6]),
            'score': score,
        }
        score_list.append(pred_info)
    json_data = json.dumps(score_list, indent=4)
    file = open(os.path.join(results_dir, 'score.json'), 'w')
    file.write(json_data)
    file.close()


if __name__ == "__main__":
    import numpy as np
    from collections import OrderedDict
    results_file = '../../LLD-MMRI2023-main/merge_unifB_0615_0705/test_res.txt'
    outputs = np.load('../../LLD-MMRI2023-main/merge_unifB_0615_0705/preds.npy')
    targets = np.load('../../LLD-MMRI2023-main/merge_unifB_0615_0705/gts.npy')
    
    print(outputs.shape) # (n_samples, n_classes)
    print(targets.shape) # (n_samples, )
    ##### macro average metrics
    acc = ACC(outputs, targets)
    f1 = F1_score(outputs, targets)
    recall = Recall(outputs, targets)
    precision = Precision(outputs, targets)
    ##### metrics on every class
    kappa = Cohen_Kappa(outputs, targets)
    report = cls_report(outputs, targets)
    cm = confusion_matrix(outputs, targets)
    specificity = Specificity(outputs, targets)
    npv = Npv_cls(outputs, targets)
    auc = AUC(outputs, targets)
    metrics = OrderedDict([
        # ('avr_auc', np.mean(auc)),
        ('acc', acc),
        ('f1', f1),
        ('recall', recall),
        ('PPV', precision),
        ('NPV', np.mean(npv)),
        ('avr_specificity', np.mean(specificity)),
        ('kappa', kappa),
        ('confusion matrix', cm),
        ('classification report', report),
        ('specificity', specificity),
        ('npv', npv),
        # ('auc', auc) 
    ])
    # print(metrics)

    output_str = 'Test Results:\n'
    for key, value in metrics.items():
        if key == 'confusion matrix':
            output_str += f'{key}:\n {value}\n'
        elif key == 'classification report':
            output_str += f'{key}:\n {value}\n'
        else:
            output_str += f'{key}: {value}\n'

    file = open(results_file, 'w')
    file.write(output_str)
    file.close()
