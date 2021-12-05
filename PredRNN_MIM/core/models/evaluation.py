try:
    import cPickle as pickle
except:
    import pickle
import numpy as np
import logging
import os
from core.data_provider.dataloader import get_exclude_mask

try:
    from core.models.numba_accelerated import get_GDL_numba, get_hit_miss_counts_numba,\
        get_balancing_weights_numba
except:
    raise ImportError("Numba has not been installed correctly!")

class HKOEvaluation(object):
    def __init__(self, seq_len, no_ssim=True, threholds=None):
        self._thresholds = np.array([0.5, 2, 5, 10, 30]) if threholds is None else threholds
        self._seq_len = seq_len
        self._no_ssim = no_ssim
        self._exclude_mask = get_exclude_mask()
        self.begin()

    def begin(self):
        self._total_hits = np.zeros((self._seq_len, len(self._thresholds)), dtype=np.int)
        self._total_misses = np.zeros((self._seq_len, len(self._thresholds)),  dtype=np.int)
        self._total_false_alarms = np.zeros((self._seq_len, len(self._thresholds)), dtype=np.int)
        self._total_correct_negatives = np.zeros((self._seq_len, len(self._thresholds)),
                                                 dtype=np.int)
        self._mse = np.zeros((self._seq_len, ), dtype=np.float32)
        self._mae = np.zeros((self._seq_len, ), dtype=np.float32)
        self._balanced_mse = np.zeros((self._seq_len, ), dtype=np.float32)
        self._balanced_mae = np.zeros((self._seq_len,), dtype=np.float32)
        self._gdl = np.zeros((self._seq_len,), dtype=np.float32)
        self._ssim = np.zeros((self._seq_len,), dtype=np.float32)
        self._datetime_dict = {}
        self._total_batch_num = 0

    def clear_all(self):
        self._total_hits[:] = 0
        self._total_misses[:] = 0
        self._total_false_alarms[:] = 0
        self._total_correct_negatives[:] = 0
        self._mse[:] = 0
        self._mae[:] = 0
        self._gdl[:] = 0
        self._ssim[:] = 0
        self._total_batch_num = 0
        self._balanced_mse[:] = 0
        self._balanced_mae[:] = 0

    def update(self, gt, pred, mask, start_datetimes=None):
        """

        Parameters
        ----------
        gt : np.ndarray
        pred : np.ndarray
        mask : np.ndarray
            0 indicates not use and 1 indicates that the location will be taken into account
        start_datetimes : list
            The starting datetimes of all the testing instances

        Returns
        -------

        """
        if start_datetimes is not None:
            batch_size = len(start_datetimes)
            assert gt.shape[1] == batch_size
        else:
            batch_size = gt.shape[1]
        #assert gt.shape[0] == self._seq_len
        #assert gt.shape == pred.shape
        #assert gt.shape == mask.shape

        self._total_batch_num += batch_size
        #TODO Save all the mse, mae, gdl, hits, misses, false_alarms and correct_negatives
        mse = (mask * np.square(pred - gt)).sum(axis=(2, 3, 4))
        mae = (mask * np.abs(pred - gt)).sum(axis=(2, 3, 4))
        weights = get_balancing_weights_numba(data=gt, mask=mask,
                                              base_balancing_weights=(1, 1, 2, 5, 10, 30),
                                              thresholds=self._thresholds)
        ## <2, <5, ... 不同权值的 MSE.
        # S*B*1*H*W
        balanced_mse = (weights * np.square(pred - gt)).sum(axis=(2, 3, 4))
        balanced_mae = (weights * np.abs(pred - gt)).sum(axis=(2, 3, 4))
        gdl = get_GDL_numba(prediction=pred, truth=gt, mask=mask)
        self._mse += mse.sum(axis=1)
        self._mae += mae.sum(axis=1)
        self._balanced_mse += balanced_mse.sum(axis=1)
        self._balanced_mae += balanced_mae.sum(axis=1)
        self._gdl += gdl.sum(axis=1)
        if not self._no_ssim:
            raise NotImplementedError
            # self._ssim += get_SSIM(prediction=pred, truth=gt)
        hits, misses, false_alarms, correct_negatives = \
            get_hit_miss_counts_numba(prediction=pred, truth=gt, mask=mask,
                                      thresholds=self._thresholds)

        self._total_hits += hits.sum(axis=1)
        self._total_misses += misses.sum(axis=1)
        self._total_false_alarms += false_alarms.sum(axis=1)
        self._total_correct_negatives += correct_negatives.sum(axis=1)

    def calculate_f1_score(self):
        '''
        计算每个分段的 precision, recall 和 f1 score.
        :return:
        '''
        # a: TP, b: TN, c: FP, d: FN
        a = self._total_hits.astype(np.float64)
        b = self._total_false_alarms.astype(np.float64)
        c = self._total_misses.astype(np.float64)
        d = self._total_correct_negatives.astype(np.float64)
        precision = a / (a + c)
        recall = a / (a + d)
        return precision, recall, (2*precision*recall)/(precision+recall)

    def calculate_stat(self):
        """The following measurements will be used to measure the score of the forecaster

        See Also
        [Weather and Forecasting 2010] Equitability Revisited: Why the "Equitable Threat Score" Is Not Equitable
        http://www.wxonline.info/topics/verif2.html

        We will denote
        (a b    (hits       false alarms
         c d) =  misses   correct negatives)

        We will report the
        POD = a / (a + c)
        FAR = b / (a + b)
        CSI = a / (a + b + c)
        Heidke Skill Score (HSS) = 2(ad - bc) / ((a+c) (c+d) + (a+b)(b+d))
        Gilbert Skill Score (GSS) = HSS / (2 - HSS), also known as the Equitable Threat Score
            HSS = 2 * GSS / (GSS + 1)
        MSE = mask * (pred - gt) **2
        MAE = mask * abs(pred - gt)
        GDL = valid_mask_h * abs(gd_h(pred) - gd_h(gt)) + valid_mask_w * abs(gd_w(pred) - gd_w(gt))
        Returns
        -------

        """
        a = self._total_hits.astype(np.float64)
        b = self._total_false_alarms.astype(np.float64)
        c = self._total_misses.astype(np.float64)
        d = self._total_correct_negatives.astype(np.float64)
        pod = a / (a + c + 0.01*np.ones_like(a)) if (a+c == 0).any() else a / (a + c)
        far = a / (a + b + 0.01*np.ones_like(a)) if (a+b == 0).any() else a / (a + b)
        csi = a / (a + b + c + 0.01*np.ones_like(a)) if (a+b+c == 0).any() else a / (a + b + c)
        n = a + b + c + d
        aref = (a + b) / n * (a + c)
        gss = (a - aref) / (a + b + c - aref + 0.01*np.ones_like(a)) if\
            (a+b+c == 0).any() else (a - aref) / (a + b + c - aref)
        hss = 2 * gss / (gss + 1)
        mse = self._mse / self._total_batch_num
        mae = self._mae / self._total_batch_num
        balanced_mse = self._balanced_mse / self._total_batch_num
        balanced_mae = self._balanced_mae / self._total_batch_num
        gdl = self._gdl / self._total_batch_num
        if not self._no_ssim:
            raise NotImplementedError
            # ssim = self._ssim / self._total_batch_num
        # return pod, far, csi, hss, gss, mse, mae, gdl
        return pod, far, csi, hss, gss, mse, mae, balanced_mse, balanced_mae, gdl

    def print_stat_readable(self, prefix=""):
        logging.getLogger().setLevel(logging.INFO)
        logging.info("%sTotal Sequence Number: %d"
                     % (prefix, self._total_batch_num))
        pod, far, csi, hss, gss, mse, mae, balanced_mse, balanced_mae, gdl = self.calculate_stat()
        # pod, far, csi, hss, gss, mse, mae, gdl = self.calculate_stat()
        logging.info("   Hits: " + ', '.join([">%g:%g/%g" % (threshold,
                                                             self._total_hits[:, i].mean(),
                                                             self._total_hits[-1, i])
                                             for i, threshold in enumerate(self._thresholds)]))
        logging.info("   POD: " + ', '.join([">%g:%g/%g" % (threshold, pod[:, i].mean(), pod[-1, i])
                                  for i, threshold in enumerate(self._thresholds)]))
        logging.info("   FAR: " + ', '.join([">%g:%g/%g" % (threshold, far[:, i].mean(), far[-1, i])
                                  for i, threshold in enumerate(self._thresholds)]))
        logging.info("   CSI: " + ', '.join([">%g:%g/%g" % (threshold, csi[:, i].mean(), csi[-1, i])
                                  for i, threshold in enumerate(self._thresholds)]))
        logging.info("   GSS: " + ', '.join([">%g:%g/%g" % (threshold, gss[:, i].mean(), gss[-1, i])
                                             for i, threshold in enumerate(self._thresholds)]))
        logging.info("   HSS: " + ', '.join([">%g:%g/%g" % (threshold, hss[:, i].mean(), hss[-1, i])
                                             for i, threshold in enumerate(self._thresholds)]))
        logging.info("   MSE: %g/%g" % (mse.mean(), mse[-1]))
        logging.info("   MAE: %g/%g" % (mae.mean(), mae[-1]))
        logging.info("   Balanced MSE: %g/%g" % (balanced_mse.mean(), balanced_mse[-1]))
        logging.info("   Balanced MAE: %g/%g" % (balanced_mae.mean(), balanced_mae[-1]))
        logging.info("   GDL: %g/%g" % (gdl.mean(), gdl[-1]))
        if not self._no_ssim:
            raise NotImplementedError

    def save_pkl(self, path):
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        f = open(path, 'wb')
        logging.info("Saving HKOEvaluation to %s" %path)
        pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    def save_txt_readable(self, path):
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        pod, far, csi, hss, gss, mse, mae, balanced_mse, balanced_mae, gdl = self.calculate_stat()
        # pod, far, csi, hss, gss, mse, mae, gdl = self.calculate_stat()
        f = open(path, 'w')
        logging.info("Saving readable txt of HKOEvaluation to %s" % path)
        f.write("Total Sequence Num: %d, Out Seq Len: %d\n"
                %(self._total_batch_num,
                  self._seq_len))
        for (i, threshold) in enumerate(self._thresholds):
            f.write("Threshold = %g:\n" %threshold)
            f.write("   POD: %s\n" %str(list(pod[:, i])))
            f.write("   FAR: %s\n" % str(list(far[:, i])))
            f.write("   CSI: %s\n" % str(list(csi[:, i])))
            f.write("   GSS: %s\n" % str(list(gss[:, i])))
            f.write("   HSS: %s\n" % str(list(hss[:, i])))
            f.write("   POD stat: avg %g/final %g\n" %(pod[:, i].mean(), pod[-1, i]))
            f.write("   FAR stat: avg %g/final %g\n" %(far[:, i].mean(), far[-1, i]))
            f.write("   CSI stat: avg %g/final %g\n" %(csi[:, i].mean(), csi[-1, i]))
            f.write("   GSS stat: avg %g/final %g\n" %(gss[:, i].mean(), gss[-1, i]))
            f.write("   HSS stat: avg %g/final %g\n" % (hss[:, i].mean(), hss[-1, i]))
        f.write("MSE: %s\n" % str(list(mse)))
        f.write("MAE: %s\n" % str(list(mae)))
        f.write("Balanced MSE: %s\n" % str(list(balanced_mse)))
        f.write("Balanced MAE: %s\n" % str(list(balanced_mae)))
        f.write("GDL: %s\n" % str(list(gdl)))
        f.write("MSE stat: avg %g/final %g\n" % (mse.mean(), mse[-1]))
        f.write("MAE stat: avg %g/final %g\n" % (mae.mean(), mae[-1]))
        f.write("Balanced MSE stat: avg %g/final %g\n" % (balanced_mse.mean(), balanced_mse[-1]))
        f.write("Balanced MAE stat: avg %g/final %g\n" % (balanced_mae.mean(), balanced_mae[-1]))
        f.write("GDL stat: avg %g/final %g\n" % (gdl.mean(), gdl[-1]))
        f.close()

    def save(self, prefix):
        self.save_txt_readable(prefix + ".txt")
        self.save_pkl(prefix + ".pkl")
