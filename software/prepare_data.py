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
    inp_fea_len = inp['input_mix_feature'].shape[1]
    inp_fea_dim = inp['input_mix_feature'].shape[-1]
    inp_spec_dim = inp['input_mix_spectrum'].shape[-1]
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


# def get_spatial_audio(signal,azimuth):
#     # Listener's position and orientation
#     listener_pos = np.array([4.0, 3.0, 1.5])  # Center of the listener's head
#     distance_from_listener = 2.0  # Distance from the listener to the source
#     # Calculate the source position for the desired azimuth
#     pos_src = calculate_source_position(azimuth, distance_from_listener, listener_pos)
#     # Room dimensions [width, length, height] in meters
#     room_sz = np.array([8.0, 6.0, 6.0])
#
#     # Reflection coefficients of the walls [x0, x1, y0, y1, z0, z1] for an anechoic room
#     beta = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # adjust beta to make it anechonic
#
#     pos_rcv = np.array([[3.9, 3.0, 1.5], [4.1, 3.0, 1.5]])  # Two receivers 20 cm apart
#
#     # Number of images to simulate in each dimension for an anechoic room
#     nb_img = np.array([1, 1, 1])
#
#     # RIRs length in seconds for an anechoic room
#     # It should be at least the time it takes for sound to travel from the source to the receiver
#     c = 343.0  # Speed of sound in m/s
#     Tmax = distance_from_listener / c
#
#     # Optional parameters
#     Tdiff = None  # Time when the ISM is replaced by a diffuse reverberation model
#     spkr_pattern = "omni"
#     mic_pattern = "omni"
#     orV_src = None  # Orientation of the source
#     orV_rcv = None  # Orientation of the receiver
#     # Simulate Room Impulse Responses for both sources
#     RIRs_src = gpuRIR.simulateRIR(room_sz, beta, pos_src, pos_rcv, nb_img, Tmax, config.FRAME_RATE,
#                                     Tdiff=Tdiff, spkr_pattern=spkr_pattern, mic_pattern=mic_pattern,
#                                     orV_src=orV_src, orV_rcv=orV_rcv, c=c)
#
#     left_ear_convolved = fftconvolve(signal, RIRs_src[0][0], mode='full')
#     right_ear_convolved = fftconvolve(signal, RIRs_src[0][1], mode='full')
#
#     left_ear_convolved /= np.abs(left_ear_convolved).max()
#     right_ear_convolved /= np.abs(right_ear_convolved).max()
#
#     binaural_signal = np.vstack((left_ear_convolved, right_ear_convolved)).T
#
#     return binaural_signal
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
    batch_input_mix_fea = []
    batch_input_mix_spec = []
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
        target_spk = None
        mix_len = 0
        target_sig = None
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
                # wav_mix = signal
                target_sig = signal
                target_spk = spk_to_idx[spk]
                # sf.write('signal1_before.wav', signal, config.FRAME_RATE)
                # signal1 = get_spatial_audio(signal,config.LOC_1)
                wav_mix = signal
                # sf.write('signal1_after.wav', wav_mix, config.FRAME_RATE)
            else:
                # mixture signal
                # spatialized signal by gpu-RIR
                # input: signal, location 1 and location 2
                # if wav_mix == None --> use loc 1, else, use loc1, loc1 is always for the first speaker
                # sf.write('signal2_before.wav', signal, config.FRAME_RATE)
                # signal2 = get_spatial_audio(signal, config.LOC_2)
                # sf.write('signal2_after.wav', signal2, config.FRAME_RATE)
                wav_mix = wav_mix + signal

                # if config.EAR == 1:
                #     wav_mix = wav_mix[:,0]
                # else:
                #     wav_mix = wav_mix[:,1]
                # wav_mix /= np.abs(wav_mix).max()
                # wav_mix = wav_mix[0:len(signal)]
                # sf.write('spatial_test.wav', wav_mix, config.FRAME_RATE)

        if config.IS_LOG_SPECTRAL:
            feature_mix = np.log(np.transpose(np.abs(librosa.core.spectrum.stft(wav_mix, config.FRAME_LENGTH,
                                                                                config.FRAME_SHIFT,
                                                                                window=config.WINDOWS)))
                                 + np.spacing(1))
        else:
            feature_mix = np.transpose(np.abs(librosa.core.spectrum.stft(wav_mix, config.FRAME_LENGTH,
                                                                         config.FRAME_SHIFT,
                                                                         window=config.WINDOWS)))

        spec_mix = np.transpose(np.abs(librosa.core.spectrum.stft(wav_mix, config.FRAME_LENGTH,
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

        batch_input_mix_fea.append(feature_mix)
        batch_input_mix_spec.append(spec_mix)
        batch_input_spk.append(target_spk)
        batch_input_clean_fea.append(feature_inp_clean)
        batch_target_spec.append(spec_clean)
        batch_input_len.append(mix_len)
        batch_count += 1

        if batch_count == batch_size:
            # mix_input_fea (batch_size, time_steps, feature_dim)
            mix_input_fea = np.array(batch_input_mix_fea).reshape((batch_size, ) + feature_mix.shape)
            # mix_input_spec (batch_size, time_steps, spectrum_dim)
            mix_input_spec = np.array(batch_input_mix_spec).reshape((batch_size, ) + spec_mix.shape)
            # target_input_spk (batch_size, 1)
            target_input_spk = np.array(batch_input_spk, dtype=np.int32).reshape((batch_size, 1))
            # clean_input_fea (batch_size, time_steps, feature_dim)
            clean_input_fea = np.array(batch_input_clean_fea).reshape((batch_size, ) + feature_inp_clean.shape)
            # clean_target_spec (batch_size, time_steps, spectrum_dim)
            clean_target_spec = np.array(batch_target_spec).reshape((batch_size, ) + spec_clean.shape)

            yield ({'input_mix_feature': mix_input_fea, 'input_mix_spectrum': mix_input_spec,
                    'input_target_spk': target_input_spk, 'input_clean_feature': clean_input_fea},
                   {'target_clean_spectrum': clean_target_spec})
            batch_input_mix_fea = []
            batch_input_mix_spec = []
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
    print (x['input_mix_feature'].shape)
    print (x['input_mix_spectrum'].shape)
    print (x['input_target_spk'].shape)
    print (x['input_clean_feature'].shape)
    print (y['target_clean_spectrum'].shape)
