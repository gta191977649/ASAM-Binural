# <-*- encoding:utf-8 -*->
"""
    The main function of ASAM [AAAI 2018]
        Xu, J.; Shi, J.; Liu, G.; Chen, X.; Xu B. 2018
        Modeling attention and memory for auditory selection in a cocktail party environment.
        In Proceedings of the 32nd AAAI Conference on Artificial Intelligence.
        https://github.com/jacoxu/ASAM
"""
# import matlab
# import matlab.engine
import numpy as np
import config
import time
import nnet
import os
import matplotlib.pyplot as plt

np.random.seed(1)
__author__ = '[jacoxu](https://github.com/jacoxu)'

def main():
    config.init_config()
    print('go to model')
    print ('*' * 80)
    # _log_file_test = open(config.LOG_FILE_PRE + '_Left_' + 'test', 'aw')
    _log_file = open(config.LOG_FILE_PRE + time.strftime("_%Y_%m_%d_%H%M%S", time.localtime()), 'w')
    # _log_file_test = open(config.LOG_FILE_PRE + '_Left_' + 'test', 'w')
    # log configuration.
    config.log_config(_log_file)
    # initialize model
    weights_path = None

    # if config.MODE == 2:
    #     if config.DATASET == 'WSJ0':
    #         weights_path = './_tmp_weights/WSJ0_30_mono.h5'
    #     elif config.DATASET == 'THCHS-30':
    #         weights_path = './_tmp_weights/ASAM_THCHS30_weight_00034.h5'
    dl4ss_model = nnet.NNet(_log_file, weights_path)

    # if config.MODE == 1:
    print ('Start to train model ...')
    _log_file.write('Start to train model ...\n')
    dl4ss_model.train()
    # elif config.MODE == 3:
    #     print('Start to train the binaural Model ...\n')
    #     _log_file.write('Start to train binaural model ...\n')
    #     dl4ss_model.train_binaural()

    # print ('valid spk number: 2')
    # _log_file.write('valid spk number: 2\n')
    # dl4ss_model.predict(config.VALID_LIST, spk_num=2)
    # print ('test spk number: 2')
    # _log_file.write('test spk number: 2\n')
    # dl4ss_model.predict(config.TEST_LIST, spk_num=2)
    # print ('test spk number: 3')
    # _log_file.write('test spk number: 3\n')
    # dl4ss_model.predict(config.TEST_LIST, spk_num=3)
    # print ('test spk number: 2 with bg noise')
    # _log_file.write('test spk number: 2 with bg noise\n')
    # dl4ss_model.predict(config.TEST_LIST, spk_num=2, add_bgd_noise=True)
    #
    # for supp_time in [0.25, 0.5, 1, 2, 4, 8, 16, 32]:
    #     print ('unk spk and supplemental wav span: %02d' % supp_time)
    #     _log_file.write('unk spk and supplemental wav span: %02d\n' % supp_time)
    #     dl4ss_model.predict(config.UNK_LIST, spk_num=2, unk_spk=config.UNK_SPK, supp_time=supp_time)
    # else:
    #     print ('Wrong mode: %s' % config.MODE)
    #     _log_file.write('Wrong mode: %s\n' % config.MODE)
    _log_file.close()

if __name__ == "__main__":
    print ('Here is the main function of ASAM [AAAI 2018]')
    main()
