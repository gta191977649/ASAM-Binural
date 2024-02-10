import numpy as np
# import gpuRIR
import soundfile as sf
from scipy.signal import fftconvolve
from scipy.io.wavfile import write
import matplotlib.pyplot as plt

def calculate_source_position(azimuth_deg, distance, listener_pos):
    # Convert azimuth angle from degrees to radians
    azimuth_rad = np.radians(azimuth_deg)

    # Calculate source position in the horizontal plane (x, y)
    src_x = listener_pos[0] + distance * np.cos(azimuth_rad)
    src_y = listener_pos[1] + distance * np.sin(azimuth_rad)
    src_z = listener_pos[2]  # Keep the source at the same height as the listener

    return np.array([src_x, src_y, src_z])

# Listener's position and orientation
listener_pos = np.array([4.0, 3.0, 1.5])  # Center of the listener's head
distance_from_listener = 2.0  # Distance from the listener to the source

# Desired azimuth angle for the source
desired_azimuth_1 = 180
desired_azimuth_2 = 30

# Calculate the source position for the desired azimuth
pos_src_1 = calculate_source_position(desired_azimuth_1, distance_from_listener, listener_pos)
print("Calculated Source Position_1:", pos_src_1)
pos_src_2 = calculate_source_position(desired_azimuth_2, distance_from_listener, listener_pos)
print("Calculated Source Position_2:", pos_src_2)

# Room dimensions [width, length, height] in meters
room_sz = np.array([8.0, 6.0, 6.0])

# Reflection coefficients of the walls [x0, x1, y0, y1, z0, z1] for an anechoic room
beta = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # adjust beta to make it anechonic


pos_rcv = np.array([[3.9, 3.0, 1.5], [4.1, 3.0, 1.5]]) # Two receivers 20 cm apart

# Number of images to simulate in each dimension for an anechoic room
nb_img = np.array([1, 1, 1])


# RIRs length in seconds for an anechoic room
# It should be at least the time it takes for sound to travel from the source to the receiver
c = 343.0  # Speed of sound in m/s
Tmax = distance_from_listener / c

# Optional parameters
Tdiff = None  # Time when the ISM is replaced by a diffuse reverberation model
spkr_pattern = "omni"
mic_pattern = "omni"
orV_src = None  # Orientation of the source
orV_rcv = None  # Orientation of the receiver
# Load an audio file as the dry signal
dry_signal1, sr = sf.read('401a010a.wav')
# Load an audio file as the dry signal
dry_signal2, sr = sf.read('403a010p.wav')
# Simulate Room Impulse Responses for both sources
RIRs_src_1 = gpuRIR.simulateRIR(room_sz, beta, pos_src_1, pos_rcv, nb_img, Tmax, sr,
                                Tdiff=Tdiff, spkr_pattern=spkr_pattern, mic_pattern=mic_pattern,
                                orV_src=orV_src, orV_rcv=orV_rcv, c=c)

RIRs_src_2 = gpuRIR.simulateRIR(room_sz, beta, pos_src_2, pos_rcv, nb_img, Tmax, sr,
                                Tdiff=Tdiff, spkr_pattern=spkr_pattern, mic_pattern=mic_pattern,
                                orV_src=orV_src, orV_rcv=orV_rcv, c=c)

# Convolve dry signal with RIRs for each source
# Assuming the same dry signal is used for both sources
max_len = max(len(dry_signal1), len(dry_signal2))
if len(dry_signal1) <= len(dry_signal2):
    dry_signal1 = np.pad(dry_signal1, (0, max_len - len(dry_signal1)), mode='constant')
else:
    dry_signal2 = np.pad(dry_signal2, (0, max_len - len(dry_signal2)), mode='constant')

left_ear_convolved_1 = fftconvolve(dry_signal1, RIRs_src_1[0][0], mode='full')
right_ear_convolved_1 = fftconvolve(dry_signal1, RIRs_src_1[0][1], mode='full')
# Normalize the signals
max_val_1 = max(np.abs(left_ear_convolved_1).max(), np.abs(right_ear_convolved_1).max())
left_ear_convolved_1 /= max_val_1
right_ear_convolved_1 /= max_val_1
left_ear_convolved_2 = fftconvolve(dry_signal2, RIRs_src_2[0][0], mode='full')
right_ear_convolved_2 = fftconvolve(dry_signal2, RIRs_src_2[0][1], mode='full')
# Normalize the signals
max_val_2 = max(np.abs(left_ear_convolved_2).max(), np.abs(right_ear_convolved_2).max())
left_ear_convolved_2 /= np.abs(left_ear_convolved_2).max()
right_ear_convolved_2 /= np.abs(right_ear_convolved_2).max()
test = np.vstack((left_ear_convolved_2, right_ear_convolved_2)).T
# plot the binaural audio
plt.figure()
plt.plot(test[:,1],color='r', label='right')
plt.plot(test[:,0],color='b', label='left')
plt.legend(loc='best')
plt.xlim([25000,25500])
plt.show()
write('spatial_audio_1src_test.wav', sr, test)
# Normalize the signals
# Combine the signals from both sources
left_ear_combined = left_ear_convolved_1 + left_ear_convolved_2
right_ear_combined = right_ear_convolved_1 + right_ear_convolved_2
# Normalize the signals
max_val = max(np.abs(left_ear_combined).max(), np.abs(right_ear_combined).max())
left_ear_combined /= max_val
right_ear_combined /= max_val

binaural = np.vstack((left_ear_combined, right_ear_combined)).T
write('spatial_audio_2src_180_30.wav', sr, binaural)

# plot the binaural audio
plt.figure()
plt.plot(binaural[:,0],color='b', label='left')
plt.plot(binaural[:,1],color='r', label='right')
plt.show()

# only phase difference or introduce level difference