# Pacs

Official Pytorch implementation for the paper presented on 
WSDM 2022 titled ''*A New Class of Polynomial Activation 
Functions of Deep Learning for Precipitation Forecasting*''.

## Environment Requirements

Python 3.6+

PyTorch 1.0 and Ubuntu or macOS.

opencv-contrib-python

numba

## Dataset
For the rainfall prediction, you should apply for HKO-7 Dataset from [HKO-7](https://github.com/sxjscience/HKO-7).
For the synthetic dataset, data are generated during training.

## Train and Test

### ConvGRU
Pacs are added based on the code of [ConvRNN-pytorch](https://github.com/Hzzone/Precipitation-Nowcasting)

You should first modify file *ConvRNN/now/config.py*. 

Please modify the MODEL_SAVE_DIR and MODEL_NAME to fit your model saving path

Please modify the dirs of HKO_PNG_PATH and HKO_MASK_PATH to fit the data you downloaded.

You can change the NORM and NORM_PEND to add normalization after or before the activation.

You can change CNN_ACT_TYPE for different activation functions.

Then go to dir *ConvRNN/exp*.

If you want to train the model, modify the last line of *main.py* to `train()`.
And run `python3 main.py`.
If you want to test the model, modify the last line of *main.py* to `test()`.
And run `python3 main.py`. Make sure that model are downloaded to the path *MODEL_SAVE_DIR / MODEL_NAME*

### PredRNN and MIM
Pacs are added based on the code of [predrnn-pytorch](https://github.com/thuml/predrnn-pytorch)

Go to dir *PredRNN_MIM*.

You should first modify file *run.py*. 

Please modify the *save_dir* and *gen_frm_dir* to fit your model saving path

Please modify the dirs of *core/data_provider/dataloader.py* and *core/data_provider/image.py* to fit the data you downloaded.

You can change the *norm* to `0` for no normalization; `1` for layer normalization; `2` for range norm.

You can change activator for different activation functions.

*num_hidden* should be '64,64,64,64' for MIM and '128, 128' for PredRNN.

If you want to train the model, modify the last line of *run.py* to `train_()`.
And run `python3 run.py`.
If you want to test the model, modify the last line of *main.py* to `test_()`.
And run `python3 run.py`. Make sure that model are downloaded to the path *MODEL_SAVE_DIR / MODEL_NAME*

### Synthetic dataset
You can modify `order_of_poly` in *synthetic.py* to approximate polynomials of different orders.

Run `python3 synthetic.py`, you can get an *.xls* file for results.
