import numpy as np
import mir_eval
import config
def bss_eval(wav_truth_signal, wav_truth_noise, wav_pred_signal, mixture):
    # Stack the reference signals
    reference_signals = np.vstack([wav_truth_signal, wav_truth_noise])

    # Adding a small amount of noise to avoid all-zero array
    noise_floor = 1e-10 * np.random.randn(*wav_truth_noise.shape)

    # Stack the predicted signal with noise_floor to match the shape
    estimated_signal = np.vstack([wav_pred_signal, noise_floor])

    # Evaluate the predicted signal
    sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(reference_signals,  estimated_signal,
                                                            compute_permutation=True)
    sdr2, sir2, sar2, _ = mir_eval.separation.bss_eval_sources(wav_truth_signal, mixture,
                                                            compute_permutation=True)

    # Compute SDR improvement, which is also known as NSDR (Normalized Source-to-Distortion Ratio)
    # sdr[0] corresponds to the SDR of the target (clean source), and sdr[1] corresponds to the SDR of the noise
    # The improvement in SDR for the target signal is of interest, hence NSDR = SDR_improvement for target signal
    # sdr_improvement = sdr[0] - sdr[1]  # Assuming sdr[0] is the target and sdr[1] is the noise
    nsdr = sdr[0]- sdr2 # NSDR = SDR_after - SDR_before for the target signal

    parms = {
        'SDR': sdr[0],
        'SIR': sir[0],
        'SAR': sar[0],
        'NSDR': nsdr
    }

    return parms

def metrics_log_write(text):
    _log_file_test = open(config.LOG_FILE_PRE + '_Mono_' + 'test', 'a')
    _log_file_test.write(text)
    _log_file_test.close()