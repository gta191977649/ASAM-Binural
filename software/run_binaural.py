import numpy as np
import config
import time
import BNet
np.random.seed(1)

def main():
    config.init_config()
    print('go to model')
    print('*' * 80)
    _log_file = open(config.LOG_FILE_PRE + time.strftime("_%Y_%m_%d_%H%M%S", time.localtime()), 'w')
    config.log_config(_log_file)
    # initialize model
    weights_path = None
    dl4ss_model = BNet.BNet_Mag(_log_file, weights_path)
    print('Start to train model ...')
    _log_file.write('Start to train model ...\n')
    print(dl4ss_model.auditory_model.get_config())
    dl4ss_model.train()
    _log_file.close()

if __name__ == "__main__":
    print('Binaural_Model')
    main()
