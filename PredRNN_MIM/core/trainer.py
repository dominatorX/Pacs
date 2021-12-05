import os.path
import datetime
import cv2
import shutil
import numpy as np
import logging
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


def train(model, ims, real_input_flag, mask, configs, itr):
    cost = model.train(ims, real_input_flag, mask)
    if configs.reverse_input:
        ims_rev = np.flip(ims, axis=0).copy()
        cost += model.train(ims_rev, real_input_flag, mask)
        cost = cost / 2

    if itr % configs.display_interval == 0:
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'itr: '+str(itr))
        print('training loss: ' + str(cost))


def test(model, test_input_handle, configs, itr, rid):
    # print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S ')+str(rid)+' test...')
    logger.info(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S ')+str(rid)+' test...')
    res_path = os.path.join(configs.gen_frm_dir, str(rid))
    res_path = os.path.join(res_path, str(itr))
    if os.path.exists(res_path):
        shutil.rmtree(res_path)
    os.mkdir(res_path)
    test_loss = 0.0
    test_mse = 0.0
    test_mae = 0.0
    batch_id = 0
    gen_id = 1

    real_input_flag = np.zeros(
        (configs.batch_size,
         configs.total_length - configs.input_length - 1,
         (configs.img_height - 1) // configs.patch_size + 1,
         (configs.img_width - 1) // configs.patch_size + 1,
         configs.patch_size ** 2 * configs.img_channel))

    while not test_input_handle.use_up:
        batch_id += 1
        test_dat, test_mask, sample_datetimes, _ = \
            test_input_handle.sample(batch_size=configs.batch_size)

        if test_dat.shape[1] < configs.batch_size:
            break

        img_gen, loss, mse, mae = model.test(test_dat, real_input_flag, test_mask)
        test_mse += mse
        test_mae += mae
        test_loss += loss
        img_gen_length = img_gen.shape[0]

        # save prediction examples
        if gen_id <= configs.num_save_samples:
            for idx in range(configs.batch_size):
                if gen_id > configs.num_save_samples:
                    break
                path = os.path.join(res_path, str(gen_id))
                os.mkdir(path)
                for i in range(configs.total_length):
                    name = 'gt' + str(i + 1) + '.png'
                    file_name = os.path.join(path, name)
                    img_gt = np.uint8(test_dat[i, idx, -1, :, :] * 255)[:, :, np.newaxis]
                    cv2.imwrite(file_name, img_gt)
                for i in range(img_gen_length):
                    name = 'pd' + str(i + 1) + '.png'
                    file_name = os.path.join(path, name)
                    img_pd = img_gen[i, idx, 0, :, :][:, :, np.newaxis]
                    img_pd = np.maximum(img_pd, 0)
                    img_pd = np.minimum(img_pd, 1)
                    img_pd = np.uint8(img_pd * 255)
                    cv2.imwrite(file_name, img_pd)
                gen_id += 1

    _, _, valid_csi, valid_hss, _, valid_mse, valid_mae, valid_balanced_mse, valid_balanced_mae, _ = \
        model.evaluator.calculate_stat()

    test_mse /= batch_id
    test_mae /= batch_id

    logger.info("valid: csi "+str(valid_csi)+" valid: hss "+str(valid_hss) +
                " bmse "+str(test_mse)+" bmae "+str(test_mae))
    test_loss /= batch_id
    logger.info('loss: '+str(test_loss))
    return float(test_loss)

