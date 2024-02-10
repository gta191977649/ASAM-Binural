# -*- coding: utf8 -*-

import time
import config
from keras.layers import Input, Bidirectional, LSTM, TimeDistributed, Dense, merge, Reshape, Embedding, Masking
from keras.models import Model
from keras.optimizers import SGD, Nadam
import keras.backend as K
from extend_layers import Attention, MeanPool, SpkLifeLongMemory, SelectSpkMemory, update_memory, MaskingGt
from model_init_nnet import ModelInitNNet
import predict
import numpy as np
import matplotlib.pyplot as plt
import h5py
__author__ = '[jacoxu](https://github.com/jacoxu)'


class NNet(ModelInitNNet):
    def __init__(self, _log_file, weights_path):
        # load data during initialization
        super(NNet, self).__init__(_log_file)
        self.log_file = _log_file
        # self.optimizer = SGD(lr=0.05, decay=0, momentum=0.9, nesterov=True)
        self.optimizer = Nadam(clipnorm=200)
        print("Start to build models")
        self.auditory_model, self.spk_memory_model = self.build_models(weights_path)
        print("Finished models building")
    def build_models(self, weights_path=None):
        # inp_mix_fea_shape (MaxLen(time), feature_dim)
        mix_fea_inp = Input(shape=(self.inp_fea_len, self.inp_fea_dim), name='input_mix_feature')
        # inp_mix_spec_shape (MaxLen(time), spectrum_dim), fix the time_steps here
        mix_spec_inp = Input(shape=(self.inp_fea_len, self.inp_spec_dim), name='input_mix_spectrum')
        # inp_target_spk_shape (1)
        target_spk_inp = Input(shape=(self.inp_spk_len, ), name='input_target_spk')
        # inp_clean_fea_shape (MaxLen(time), feature_dim)，not fix the time_steps
        clean_fea_inp = Input(shape=(None, self.inp_fea_dim), name='input_clean_feature')

        mix_fea_layer = mix_fea_inp
        mix_spec_layer = mix_spec_inp
        target_spk_layer = target_spk_inp
        if config.IS_LOG_SPECTRAL:
            clean_fea_layer = MaskingGt(mask_value=np.log(np.spacing(1) * 2))(clean_fea_inp)
        else:
            clean_fea_layer = Masking(mask_value=0.)(clean_fea_inp)
        # biLSTM to extract the speech features
        # (None(batch), MaxLen(time), feature_dim) -> (None(batch), None(time), hidden_dim)
        for _layer in range(config.NUM_LAYERS):
            mix_fea_layer = \
                Bidirectional(LSTM(config.HIDDEN_UNITS, return_sequences=True),
                              merge_mode='concat')(mix_fea_layer)
        # (None(batch), MaxLen(time), hidden_dim) -> (None(batch), MaxLen(time), spec_dim * embed_dim)
        mix_embedding_layer = TimeDistributed(Dense(self.inp_spec_dim*config.EMBEDDING_SIZE
                                                    , activation='tanh'))(mix_fea_layer)

        # (None(batch), MaxLen(time), spec_dim * embed_dim) -> (None(batch), MaxLen(time), spec_dim, embed_dim)
        mix_embedding_layer = Reshape((self.inp_fea_len, self.inp_spec_dim, config.EMBEDDING_SIZE))(mix_embedding_layer)

        # extract the clean speech feature of target speaker, while the input would be fixed to zeros during test.
        # (None(batch), MaxLen(time), feature_dim) -> (None(batch), MaxLen(time), embed_dim)
        for _layer in range(config.NUM_LAYERS):
            clean_fea_layer = Bidirectional(LSTM(config.EMBEDDING_SIZE//2, return_sequences=True),
                                            merge_mode='concat')(clean_fea_layer)
        # Pooling
        # (None(batch), MaxLen(time), embed_dim) -> (None(batch), embed_dim)
        spk_vector_layer = MeanPool(name='MeanPool')(clean_fea_layer)
        # update to life-long memory
        # [((None(batch), 1), ((None(batch), embed_dim))] -> (None(batch), spk_size, embed_dim)
        spk_life_long_memory_layer = SpkLifeLongMemory(self.spk_size, config.EMBEDDING_SIZE, unk_spk=config.UNK_SPK,
                                                       name='SpkLifeLongMemory')([target_spk_layer, spk_vector_layer])

        # extract the memory corresponding to current batch input speakers
        # (None(batch), embed_dim)
        spk_memory_layer = SelectSpkMemory(name='SelectSpkMemory')([target_spk_layer, spk_life_long_memory_layer])

        # Attention(Masking)
        # (None(batch), MaxLen(time), spec_dim)
        # Adding the spatial MTPC feature into the attention block
        output_mask_layer = Attention(self.inp_fea_len, self.inp_spec_dim, config.EMBEDDING_SIZE,
                                      mode='align', name='Attention')([mix_embedding_layer
                                                                       , spk_memory_layer])
        # plot_and_save_attention_mask(output_mask_layer, save_path='attention_mask.png', plot_title='Attention Mask')
        # masking
        # (None(batch), MaxLen(time), spec_dim)
        output_clean = merge([output_mask_layer, mix_spec_layer], mode='mul', name='target_clean_spectrum')

        auditory_model = Model(input=[mix_fea_inp, mix_spec_inp, target_spk_inp, clean_fea_inp],
                               output=[output_clean], name='auditory_model')

        # output memory for updating outside.
        spk_memory_model = Model(input=auditory_model.input,
                                 output=auditory_model.get_layer('SelectSpkMemory').output,
                                 name='spk_vec_model')

        # Use the safe_load_weights function in your build_models method
        # if weights_path:
        #     print('Load the trained weights of ', weights_path)
        #     self.log_file.write('Load the trained weights of %s\n' % weights_path)
        #     self.safe_load_weights(auditory_model, weights_path)

        if weights_path:
            print ('Load the trained weights of ', weights_path)
            self.log_file.write('Load the trained weights of %s\n' % weights_path)
            # auditory_model.load_weights(weights_path)
            # self.safe_load_weights(auditory_model, weights_path)
            if weights_path:
                print('Load the trained weights of ', weights_path)
                self.log_file.write('Load the trained weights of %s\n' % weights_path)

                # Open the HDF5 file
                with h5py.File(weights_path, 'r') as f:
                    # Decode layer names only if they are byte strings
                    layer_names = [n.decode('utf8') if isinstance(n, bytes) else n for n in f.attrs['layer_names']]

                    # Load weights for each layer
                    for name in layer_names:
                        g = f[name]
                        weights = [g[weight_name] for weight_name in g.attrs['weight_names']]
                        auditory_model.get_layer(name).set_weights(weights)
            # print(auditory_model.load_weights(weights_path))
            # print('finished')
            # auditory_model.output_mask_layer()
            # plot_and_save_attention_mask(output_mask_layer, save_path='attention_mask.png', plot_title='Attention Mask')
        print ('Compiling...')
        time_start = time.time()

        auditory_model.compile(loss='mse', optimizer=self.optimizer)
        time_end = time.time()
        print ('Compiled, cost time: %f second' % (time_end - time_start))

        print("SUMMARY auditory_model")
        auditory_model.summary()
        print("SUMMARY spk_memory_model")
        spk_memory_model.summary()
        return auditory_model, spk_memory_model

    def safe_load_weights(self, model, weights_path):
        try:
            model.load_weights(weights_path)
        except AttributeError as error:
            if "'str' object has no attribute 'decode'" in str(error):
                # Handle the case where the layer names are already in string format
                with h5py.File(weights_path, 'r') as f:
                    # Load layer names as strings directly
                    layer_names = [n for n in f.attrs['layer_names']]
                # Manually load weights for each layer
                for name in layer_names:
                    g = f[name]
                    weights = [g[weight] for weight in g.attrs['weight_names']]
                    model.get_layer(name).set_weights(weights)
            else:
                # If the error is not what we expect, raise it
                raise

    def get_model_structure(self):
        """
        Print the structure of the auditory model.
        """
        self.auditory_model.summary()

    def save_entire_model(self, filepath='auditory_model.h5'):
        """
        Save the entire auditory model.
        :param filepath: Path to save the model.
        """
        self.auditory_model.save(filepath)
        print(f"Model saved to {filepath}")

    def get_attention_layer_output(self, inputs):
        """
        Get the output of the attention layer.
        :param inputs: Dictionary containing model inputs.
        :return: Output of the attention layer.
        """
        # Create a new model that outputs the attention layer output
        attention_layer_model = Model(inputs=self.auditory_model.input,
                                      outputs=self.auditory_model.get_layer('Attention').output)
        
        # Predict the attention layer output
        return attention_layer_model.predict(inputs)
    def plot_and_save_attention_mask(attention_mask, save_path='attention_mask.png', plot_title='Attention Mask'):
        """
        Plot and save the attention mask as an image file.

        :param attention_mask: numpy array representing the attention mask.
        :param save_path: string, path where the image will be saved.
        :param plot_title: string, title of the plot.
        """
        if not isinstance(attention_mask, np.ndarray):
            raise ValueError("Attention mask must be a numpy array")

        # Plotting
        plt.figure(figsize=(10, 4))
        plt.imshow(attention_mask, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar()
        plt.title(plot_title)
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        
        # Save the plot as an image file
        plt.savefig(save_path)
        print(f"Saved attention mask plot to {save_path}")

        # Optionally show the plot
        plt.show()
    def train(self):
        start_ealy_stop = config.START_EALY_STOP
        lowest_dev_loss = float('inf')
        lowest_dev_epoch = start_ealy_stop
        for epoch_num in range(config.MAX_EPOCH):
            # if (epoch_num != 0) and (epoch_num < 60) and (epoch_num % 10 == 0):
            #     # learning rate decay for SGD
            #     K.set_value(self.optimizer.lr, 0.5 * K.get_value(self.optimizer.lr))
            time_start = time.time()
            loss = 0.0
            for batch_size in range(config.EPOCH_SIZE):
                inp, out = next(self.train_gen)
                loss += self.auditory_model.train_on_batch(inp, out)
                spk_memory = self.spk_memory_model.predict(inp)
                target_spk = inp['input_target_spk']
                update_memory(self.auditory_model, target_spk, spk_memory)
                time_end = time.time()
                if batch_size != config.EPOCH_SIZE-1:  
                    print ('\rCurrent batch:' + str(batch_size+1) + '/' + str(config.EPOCH_SIZE) + \
                        ', epoch:' + str(epoch_num+1) + '/' + str(config.MAX_EPOCH) + \
                        ' and loss: %.4f, cost time: %.4f sec.' % (loss, (time_end - time_start)),)
                else:
                    print ('\rCurrent batch:' + str(batch_size+1) + '/' + str(config.EPOCH_SIZE) + \
                        ', epoch:' + str(epoch_num+1) + '/' + str(config.MAX_EPOCH) + \
                        ' and loss: %.4f, cost time: %.4f sec.' % (loss, (time_end - time_start)))
                    self.log_file.write('Current batch:' + str(batch_size+1) + '/' + str(config.EPOCH_SIZE) +
                                        ' and epoch:' + str(epoch_num+1) + '/' + str(config.MAX_EPOCH) +
                                        ' and loss: %.4f, cost time: %.4f sec.\n' % (loss, (time_end - time_start)))

            if epoch_num % 1 == 0:
                # dev loss
                dev_loss = predict.eval_loss(self.auditory_model, config.VALID_LIST, 'valid', epoch_num=epoch_num,
                                             log_file=self.log_file, spk_to_idx=self.spk_to_idx,
                                             batch_size=config.BATCH_SIZE_EVAL, unk_spk=config.UNK_SPK)

                # save the parameters of the model
                if epoch_num >= start_ealy_stop:
                    if dev_loss < lowest_dev_loss:
                        lowest_dev_loss = dev_loss
                        lowest_dev_epoch = epoch_num
                        print('-----save----')
                        tmp_weight_path = config.TMP_WEIGHT_FOLDER + "/"+config.DATASET+"_weight_%05d.h5" \
                                                                                        % (epoch_num+1)
                        print(tmp_weight_path)
                        
                        self.auditory_model.save_weights(tmp_weight_path)
                    if (epoch_num - lowest_dev_epoch >= 10) or (epoch_num == config.MAX_EPOCH-1):
                        tmp_weight_path = config.TMP_WEIGHT_FOLDER + "/"+config.DATASET+"_weight_%05d.h5" \
                                                                                        % (lowest_dev_epoch+1)
                        print('-----save2----')
                        print(tmp_weight_path)
                        self.auditory_model.load_weights(tmp_weight_path)
                        print('Early stop at epoch: %05d' % (lowest_dev_epoch+1))
                        self.log_file.write(('Early stop at epoch: %05d\n' % (lowest_dev_epoch+1)))
                        return
    def train_binaural(self):
        start_ealy_stop = config.START_EALY_STOP
        lowest_dev_loss = float('inf')
        lowest_dev_epoch = start_ealy_stop
        for epoch_num in range(config.MAX_EPOCH):
            # if (epoch_num != 0) and (epoch_num < 60) and (epoch_num % 10 == 0):
            #     # learning rate decay for SGD
            #     K.set_value(self.optimizer.lr, 0.5 * K.get_value(self.optimizer.lr))
            time_start = time.time()
            loss = 0.0
            for batch_size in range(config.EPOCH_SIZE):
                inp, out = next(self.train_gen)
                loss += self.auditory_model.train_on_batch(inp, out)
                spk_memory = self.spk_memory_model.predict(inp)
                target_spk = inp['input_target_spk']
                update_memory(self.auditory_model, target_spk, spk_memory)
                time_end = time.time()
                if batch_size != config.EPOCH_SIZE - 1:
                    print('\rCurrent batch:' + str(batch_size + 1) + '/' + str(config.EPOCH_SIZE) + \
                          ', epoch:' + str(epoch_num + 1) + '/' + str(config.MAX_EPOCH) + \
                          ' and loss: %.4f, cost time: %.4f sec.' % (loss, (time_end - time_start)), )
                else:
                    print('\rCurrent batch:' + str(batch_size + 1) + '/' + str(config.EPOCH_SIZE) + \
                          ', epoch:' + str(epoch_num + 1) + '/' + str(config.MAX_EPOCH) + \
                          ' and loss: %.4f, cost time: %.4f sec.' % (loss, (time_end - time_start)))
                    self.log_file.write('Current batch:' + str(batch_size + 1) + '/' + str(config.EPOCH_SIZE) +
                                        ' and epoch:' + str(epoch_num + 1) + '/' + str(config.MAX_EPOCH) +
                                        ' and loss: %.4f, cost time: %.4f sec.\n' % (loss, (time_end - time_start)))

            if epoch_num % 1 == 0:
                # dev loss
                dev_loss = predict.eval_loss(self.auditory_model, config.VALID_LIST, 'valid', epoch_num=epoch_num,
                                             log_file=self.log_file, spk_to_idx=self.spk_to_idx,
                                             batch_size=config.BATCH_SIZE_EVAL, unk_spk=config.UNK_SPK)

                # save the parameters of the model
                if epoch_num >= start_ealy_stop:
                    if dev_loss < lowest_dev_loss:
                        lowest_dev_loss = dev_loss
                        lowest_dev_epoch = epoch_num
                        tmp_weight_path = config.TMP_WEIGHT_FOLDER + "/" + config.DATASET + "_"+ config.EAR+ "_weight_%05d.h5" \
                                          % (epoch_num + 1)
                        self.auditory_model.save_weights(tmp_weight_path)
                    if (epoch_num - lowest_dev_epoch >= 10) or (epoch_num == config.MAX_EPOCH - 1):
                        tmp_weight_path = config.TMP_WEIGHT_FOLDER + "/" + config.DATASET +"_"+ config.EAR+ "_weight_%05d.h5" \
                                          % (lowest_dev_epoch + 1)
                        self.auditory_model.load_weights(tmp_weight_path)
                        print('Early stop at epoch: %05d' % (lowest_dev_epoch + 1))
                        self.log_file.write(('Early stop at epoch: %05d\n' % (lowest_dev_epoch + 1)))
                        return
    def predict(self, file_list, spk_num=3, unk_spk=True, supp_time=1, add_bgd_noise=False):
        predict.eval_separation(self.auditory_model, file_list, 'pred', epoch_num=0,
                                log_file=self.log_file, spk_to_idx=self.spk_to_idx,
                                batch_size=1, spk_num=spk_num, unk_spk=config.UNK_SPK, supp_time=supp_time
                                , add_bgd_noise=add_bgd_noise)
