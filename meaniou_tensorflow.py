import os
import csv

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
import tensorflow as tf
# from tensorflow.compat.v1.keras import backend as K
from tensorflow.keras import backend as K

iou_result = []
iou_result.append(["index(filename)", "np_result", "tf_result"])
np_iou_results = []
tf_iou_results = []


# write numpy_iou result to csv file
# index(filename)   np_result       tf_result
# ...               ...             ...
# total             np_total        tf_total
def write_result_to_file(filename):
    global tf_iou_results
    np.savetxt(filename, iou_result, delimiter=",")


## IOU in pure numpy
def numpy_iou(y_true, y_pred, n_class=2):
    global np_iou_results
    def iou(y_true, y_pred, n_class):
        # IOU = TP/(TP+FN+FP)
        IOU = []
        for c in range(n_class):
            TP = np.sum((y_true == c) & (y_pred == c))
            FP = np.sum((y_true != c) & (y_pred == c))
            FN = np.sum((y_true == c) & (y_pred != c))

            n = TP
            d = float(TP + FP + FN + 1e-12)

            iou = np.divide(n, d)
            IOU.append(iou)

        return np.mean(IOU)

    batch = y_true.shape[0]
    y_true = np.reshape(y_true, (batch, -1))
    y_pred = np.reshape(y_pred, (batch, -1))

    score = []
    for idx in range(batch):
        iou_value = iou(y_true[idx], y_pred[idx], n_class)
        score.append(iou_value)
        np_iou_results.append(iou_value)
    return np.mean(score)


## Calculating IOU across a range of thresholds, then we will mean all the
## values of IOU's.
## this function can be used as keras metrics
def numpy_mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.5):
        y_pred_ = tf.cast(y_pred > t, tf.int32)
        score = tf.py_function(numpy_iou, [y_true, y_pred_], tf.float64)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


def tf_mean_iou(y_true, y_pred):
    global tf_iou_results
    prec = []
    for t in np.arange(0.5, 1.0, 0.5):
        y_pred_ = tf.cast(y_pred > t, tf.int32)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        prec.append(score)
    val = K.mean(K.stack(prec), axis=0)
    return [val, up_opt]

def tf_mean_iou_for_single_instance(y_true, y_pred):
    global tf_iou_results
    for t in np.arange(0.5, 1.0, 0.5):
        y_pred_ = tf.cast(y_pred > t, tf.int32)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        prec.append(score)
    val = K.mean(K.stack(prec), axis=0)
    return val

if __name__ == "__main__":
    ## Seeding
    tf.compat.v1.random.set_random_seed(1234)

    ## Defining Placeholders
    shape = [1, 720, 1280, 1]
    y_true = tf.placeholder(tf.int32, shape=shape)
    y_pred = tf.placeholder(tf.int32, shape=shape)

    ## Reading the masks from the path
    for idx, path in enumerate(os.listdir("target/")):              # 어차피 target이랑 test랑 파일명 같아
        y_true_masks = np.zeros((1, 720, 1280, 1), dtype=np.int32)    # 직접 레이블링 한거
        y_pred_masks = np.zeros((1, 720, 1280, 1), dtype=np.int32)    # detect 된거
        mask = cv2.imread("target/" + path, -1)
        mask = np.expand_dims(mask, axis=-1)
        # TODO: current masking process takes a lot of time.
        # This process need to be deleted, by changing mean_iou function.
        for i in range(720):
            for j in range(1280):
                if mask[i][j] != 0:
                    mask[i][j] = 1
        y_true_masks[0] = mask
        mask = cv2.imread("test/" + path, -1)
        mask = np.expand_dims(mask, axis=-1)
        for i in range(720):
            for j in range(1280):
                if mask[i][j] != 0:
                    mask[i][j] = 1
        y_pred_masks[0] = mask

        ## Session
        with tf.compat.v1.Session() as sess:
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())

            tf_miou, conf = tf_mean_iou(y_true, y_pred)
            sess.run(conf, feed_dict={y_true: y_true_masks, y_pred: y_pred_masks})
            tf_miou = sess.run(tf_miou, feed_dict={y_true: y_true_masks, y_pred: y_pred_masks})
            print("TF mIOU: ", tf_miou)
            tf_iou_results.append(tf_miou)

    write_result_to_file("results/tf_results.csv")