# <-*- encoding: utf-8 -*->
"""
    pre-process data
"""
import numpy as np
import random
import config
import soundfile as sf
import resampy
from scipy.signal import fftconvolve
import librosa
import os
import argparse
import scipy.signal

def get_idx(train_list, valid_list=None, test_list=None):
    spk_set = set()
    audio_path_list = []
    if train_list is not None:
        audio_path_list.append(train_list)
    else:
        raise Exception("Error, train_list should not be None.")
    if valid_list is not None:
        audio_path_list.append(valid_list)
    if test_list is not None:
        audio_path_list.append(test_list)

    for audio_list in audio_path_list:
        file_list = open(audio_list)
        for line in file_list:
            line = line.strip().split()
            if len(line) < 2:
                print ('Wrong audio list file record in the line:', line)
                continue
            spk = line[-1]
            spk_set.add(spk)
        file_list.close()
    spk_to_idx = {}
    # convert the speaker to number: spk01 --> 1
    for spk in spk_set:
        spk_to_idx[spk] = int(spk[-2:])
    idx_to_spk = {}
    for spk, idx in spk_to_idx.items():  # Changed from iteritems() to items()
        idx_to_spk[idx] = spk
    return spk_to_idx, idx_to_spk


def get_dims(generator):
    inp, out = next(generator)
    inp_fea_len = inp['input_mix_feature_left'].shape[1]
    inp_fea_dim = inp['input_mix_feature_left'].shape[-1]
    inp_spec_dim = inp['input_mix_spectrum_left'].shape[-1]
    inp_spk_len = inp['input_target_spk'].shape[-1]
    out_spec_dim = out['target_clean_spectrum'].shape[-1]
    return inp_fea_len, inp_fea_dim, inp_spec_dim, inp_spk_len, out_spec_dim

def calculate_source_position(azimuth_deg, distance, listener_pos):
    # Convert azimuth angle from degrees to radians
    azimuth_rad = np.radians(azimuth_deg)

    # Calculate source position in the horizontal plane (x, y)
    src_x = listener_pos[0] + distance * np.cos(azimuth_rad)
    src_y = listener_pos[1] + distance * np.sin(azimuth_rad)
    src_z = listener_pos[2]  # Keep the source at the same height as the listener

    return np.array([src_x, src_y, src_z])

def filterHRTF(in_signal, fs, azim, room):
    rootHRFT = os.path.join(config.HRTF_DIR,config.ROOM,'16kHz')
    # read corrseponding HRTF file
    if room == 'Anechoic':
        nameHRTF = f'CortexBRIR_0s_{azim}deg_16k.wav'
    elif room == 'ROOM_A':
        nameHRTF = f'CortexBRIR_32s_{azim}deg_16k.wav'
    elif room == 'ROOM_B':
        nameHRTF = f'CortexBRIR_47s_{azim}deg_16k.wav'
    elif room == 'ROOM_C':
        nameHRTF = f'CortexBRIR_68s_{azim}deg_16k.wav'
    elif room == 'ROOM_D':
        nameHRTF = f'CortexBRIR_89s_{azim}deg_16k.wav'
    else:
        raise ValueError(f'HRTF selection not implemented for room {room}')

    # load HRTF
    hrtf, fsRef = sf.read(os.path.join(rootHRFT,nameHRTF))

    # resample if necessary
    if fs != fsRef:
        hrtf = scipy.signal.resample(hrtf,int(len(hrtf)*fs/fsRef))
    # apply HRTF via convolution (fft based)
    nChanHRFT = hrtf.shape[1]
    bin = np.zeros((len(in_signal), nChanHRFT))

    for ii in range(nChanHRFT):
        bin[:, ii] = scipy.signal.fftconvolve(in_signal, hrtf[:, ii], mode='same')

    return bin

def spatializeAudio(signal, fs, azim, room):
    nSamples, nSources = signal.shape
    binaural = np.zeros((nSamples, 2))
    for ii in range(nSources):
        binaural += filterHRTF(signal[:,ii], fs, azim[0][ii],room)
    return binaural
def get_feature(audio_list, spk_to_idx, min_mix=2, max_mix=2, batch_size=1):
    """
    :param audio_list: audio file list
        path/to/1st.wav spk1
        path/to/2nd.wav spk2
        path/to/3rd.wav spk1
    :param spk_to_idx: dict, spk1:0, spk2:1, ...
    :param min_mix:
    :param max_mix:
    :param batch_size:
    :return:
    """
    speaker_audios = {}

    batch_input_mix_fea_L = []
    batch_input_mix_fea_R = []
    batch_input_mix_spec_L = []
    batch_input_mix_spec_R = []
    batch_input_spk = []
    batch_input_clean_fea = []
    batch_target_spec = []
    batch_input_len = []
    batch_count = 0
    while True:
        mix_k = np.random.randint(min_mix, max_mix+1)

        if mix_k > len(speaker_audios):
            speaker_audios = {}
            file_list = open(audio_list)
            for line in file_list:
                line = line.strip().split()
                if len(line) != 2:
                    print ('Wrong audio list file record in the line:', line)
                    continue
                file_str, spk = line
                if spk not in speaker_audios:
                    speaker_audios[spk] = []
                speaker_audios[spk].append(file_str)
            file_list.close()

            for spk in speaker_audios:
                random.shuffle(speaker_audios[spk])

        wav_mix = None
        signal1 = None
        target_spk = None
        mix_len = 0
        target_sig = None
        # randomly select parameters for spatialize audio
        angles = range(-90, 90, 5)
        spk_angle = random.choice(angles)
        interf_angle = random.choice(angles)
        select_angles = np.column_stack((spk_angle, interf_angle))
        room = config.ROOM # add room conditions to config
        # mix speakers for creating mixture input, randomly sample 2 speaker from the 10 speakers
        for spk in random.sample(speaker_audios.keys(), mix_k):
            file_str = speaker_audios[spk].pop() # exclude the target speaker from the directory list
            if not speaker_audios[spk]:
                del(speaker_audios[spk])
            # read the target speaker's file
            signal, rate = sf.read(file_str)
            # for two channels, select one channel
            if len(signal.shape) > 1:
                signal = signal[:, 0]
           # if it is not sampled at 8k, resample to target sampling rate
            if rate != config.FRAME_RATE:
                signal = resampy.resample(signal, rate, config.FRAME_RATE, filter='kaiser_best')
            signal = list(signal)
            if len(signal) > config.MAX_LEN:
                signal = signal[:config.MAX_LEN]
            if len(signal) > mix_len:
                mix_len = len(signal)

            signal = np.array(signal)
            signal -= np.mean(signal)
            signal /= np.max(np.abs(signal)) # normalized

            signal = list(signal)

            if config.AUGMENT_DATA:
                random_shift = random.sample(range(len(signal)), 1)[0]
                signal = signal[random_shift:] + signal[:random_shift]

            if len(signal) < config.MAX_LEN:
                signal.extend(np.zeros(config.MAX_LEN - len(signal)))

            signal = np.array(signal)

            if wav_mix is None:
                target_sig = signal
                target_spk = spk_to_idx[spk]
                # sf.write('signal1_before.wav', signal, config.FRAME_RATE)
                signal1 = target_sig
                wav_mix = signal
                # sf.write('signal1_after.wav', wav_mix, config.FRAME_RATE)
            else:
                # mixture signal
                signal2 = signal
                # sf.write('signal2_after.wav', signal2, config.FRAME_RATE)
                y_binaural = np.column_stack((signal1, signal2))
                wav_mix = spatializeAudio(y_binaural, config.FRAME_RATE, select_angles, room)

        if config.IS_LOG_SPECTRAL:
            feature_mix_left = np.log(np.transpose(np.abs(librosa.core.spectrum.stft(wav_mix[:,0], config.FRAME_LENGTH,
                                                                                config.FRAME_SHIFT,
                                                                                window=config.WINDOWS)))
                                 + np.spacing(1))
            feature_mix_right = np.log(np.transpose(np.abs(librosa.core.spectrum.stft(wav_mix[:, 1], config.FRAME_LENGTH,
                                                                                     config.FRAME_SHIFT,
                                                                                     window=config.WINDOWS)))
                                      + np.spacing(1))
        else:
            feature_mix_left = np.transpose(np.abs(librosa.core.spectrum.stft(wav_mix[:,0], config.FRAME_LENGTH,
                                                                         config.FRAME_SHIFT,
                                                                         window=config.WINDOWS)))
            feature_mix_right = np.transpose(np.abs(librosa.core.spectrum.stft(wav_mix[:,1], config.FRAME_LENGTH,
                                                                              config.FRAME_SHIFT,
                                                                              window=config.WINDOWS)))

        spec_mix_left = np.transpose(np.abs(librosa.core.spectrum.stft(wav_mix[:,0], config.FRAME_LENGTH,
                                                                  config.FRAME_SHIFT, window=config.WINDOWS)))
        spec_mix_right = np.transpose(np.abs(librosa.core.spectrum.stft(wav_mix[:,1], config.FRAME_LENGTH,
                                                                  config.FRAME_SHIFT, window=config.WINDOWS)))

        if config.IS_LOG_SPECTRAL:
            feature_inp_clean = np.log(np.transpose(np.abs(librosa.core.spectrum.stft(target_sig, config.FRAME_LENGTH,
                                                                                      config.FRAME_SHIFT,
                                                                                      window=config.WINDOWS)))
                                       + np.spacing(1))
        else:
            feature_inp_clean = np.transpose(np.abs(librosa.core.spectrum.stft(target_sig, config.FRAME_LENGTH,
                                                                               config.FRAME_SHIFT,
                                                                               window=config.WINDOWS)))

        spec_clean = np.transpose(np.abs(librosa.core.spectrum.stft(target_sig, config.FRAME_LENGTH,
                                                                    config.FRAME_SHIFT, window=config.WINDOWS)))

        batch_input_mix_fea_L.append(feature_mix_left)
        batch_input_mix_fea_R.append(feature_mix_right)
        batch_input_mix_spec_L.append(spec_mix_left)
        batch_input_mix_spec_R.append(spec_mix_right)
        batch_input_spk.append(target_spk)
        batch_input_clean_fea.append(feature_inp_clean)
        batch_target_spec.append(spec_clean)
        batch_input_len.append(mix_len)

        batch_count += 1
        if batch_count == batch_size:
            # mix_input_fea (batch_size, time_steps, feature_dim)
            mix_input_fea_L = np.array(batch_input_mix_fea_L).reshape((batch_size, ) + feature_mix_left.shape)
            mix_input_fea_R = np.array(batch_input_mix_fea_R).reshape((batch_size,) + feature_mix_right.shape)
            # mix_input_spec (batch_size, time_steps, spectrum_dim)
            mix_input_spec_L = np.array(batch_input_mix_spec_L).reshape((batch_size, ) + spec_mix_left.shape)
            mix_input_spec_R = np.array(batch_input_mix_spec_R).reshape((batch_size,) + spec_mix_right.shape)
            # target_input_spk (batch_size, 1)
            target_input_spk = np.array(batch_input_spk, dtype=np.int32).reshape((batch_size, 1))
            # clean_input_fea (batch_size, time_steps, feature_dim)
            clean_input_fea = np.array(batch_input_clean_fea).reshape((batch_size, ) + feature_inp_clean.shape)
            # clean_target_spec (batch_size, time_steps, spectrum_dim)
            clean_target_spec = np.array(batch_target_spec).reshape((batch_size, ) + spec_clean.shape)

            yield ({'input_mix_feature_left': mix_input_fea_L,'input_mix_feature_right': mix_input_fea_R, 'input_mix_spectrum_left': mix_input_spec_L,
                    'input_mix_spectrum_right':mix_input_spec_R,'input_target_spk': target_input_spk, 'input_clean_feature': clean_input_fea},
                   {'target_clean_spectrum': clean_target_spec})
            batch_input_mix_fea_L = []
            batch_input_mix_fea_R = []
            batch_input_mix_spec_L = []
            batch_input_mix_spec_R = []
            batch_input_spk = []
            batch_input_clean_fea = []
            batch_target_spec = []
            batch_input_len = []
            batch_count = 0

if __name__ == "__main__":
    config.init_config()
    spk_to_idx, idx_to_spk = get_idx(config.TRAIN_LIST, config.VALID_LIST, config.TEST_LIST)
    x, y = next(get_feature(config.TRAIN_LIST, spk_to_idx, min_mix=config.MIN_MIX, max_mix=config.MAX_MIX,
                            batch_size=config.BATCH_SIZE))
    print (x['input_mix_feature_left'].shape)
    print(x['input_mix_feature_right'].shape)
    print (x['input_mix_spectrum_left'].shape)
    print(x['input_mix_spectrum_right'].shape)
    print (x['input_target_spk'].shape)
    print (x['input_clean_feature'].shape)
    print (y['target_clean_spectrum'].shape)
