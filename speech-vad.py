import os, sys, math
import argparse
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
    
def speech_vad(wav_file):
    fs, audio = wavfile.read(wav_file)
    audio = audio / 2.0 ** 14 # normalize in range (-2, +2)
    audio = audio - np.mean(audio)

    frame_size_ms = 25
    step_size_ms = 10
    frame_size_samples = int(fs * frame_size_ms / 1000)
    step_size_samples = int(fs * step_size_ms / 1000)
    number_of_frames = int((len(audio) - frame_size_samples) / step_size_samples)
    total_number_samples = int ((number_of_frames * step_size_ms 
                                 + (frame_size_ms - step_size_ms)) * fs / 1000)
    audio = audio[0: total_number_samples] # remove the last partial frame
    n_fft_points = 2
    while n_fft_points < frame_size_samples:
        n_fft_points = n_fft_points *2
    
    pitch_low = 100  # Hz
    pitch_high = 300 # Hz
    hz_per_bin = fs / n_fft_points
    bin0 = int(pitch_low / hz_per_bin)
    bin1 = math.ceil(pitch_high / hz_per_bin) + 1
    energy_array=[]
    for i in np.arange(number_of_frames):
        pos0 = int(i * step_size_samples) 
        pos1 = int(pos0 + frame_size_samples)
        frame = audio[pos0 : pos1]
        f_audio = abs(np.fft.fft(frame, n=n_fft_points))/float(n_fft_points)
        energy_array.append(float(np.mean(f_audio[bin0 : bin1])))
    energy_array = np.array(energy_array)
    
    WT = 500 # ms: window size, treat it as silence if more than 0.5s
    L = int(WT/step_size_ms)
    noise = 1 << 31;
    for n in range(number_of_frames - L):
        m = np.mean(energy_array[n : (n+1)*L])
        if m < noise:
            noise = m

    masks = list(range(number_of_frames))
    threshold = 6 * noise # noise level, 20*log10(10) = 20dB, 20*log10(6) = 16dB
    if number_of_frames * step_size_ms < 6*WT: # too short to find silence noise
        a = np.amin(energy_array)
        b = np.amax(energy_array)
        c = a + (b-a) * 1.0/3.0
        if threshold > c:
            threshold = c

    for i in range(number_of_frames):
        if energy_array[i] > threshold:
            masks[i] = 1
        else:
            masks[i] = 0

    # check 0s
    f0 = 0
    for f in range(number_of_frames):
        if masks[f] > 0:
            f0 = f
            break
    c = 0
    temp = []
    for fm in range(f0, number_of_frames):
        if masks[fm] > 0:
            if c == 0:
                continue
            if c > 0:
                temp.append(fm) # 2nd
                c = 0;
        else:
            if c == 0:
                temp.append(fm) # 1st
            c += 1
    for i in range(int(len(temp)/2)):
        p1 = temp[2*i+0]
        p2 = temp[2*i+1]
        if p2 - p1 < L: # not silence
            for p in range(p1, p2+1):
                masks[p] = 1
    # check 1s
    c = 0
    temp = []
    for fm in range(number_of_frames):
        if masks[fm] < 1:
            if c == 0:
                continue
            if c > 0:
                temp.append(fm) # 2nd
                c = 0;
        else:
            if c == 0:
                temp.append(fm) # 1st
            c += 1
    for i in range(int(len(temp)/2)):
        p1 = temp[2*i+0]
        p2 = temp[2*i+1]
        if p2 - p1 < int(L/5): # not active
            for p in range(p1, p2+1):
                masks[p] = 0

    vad_masks = np.repeat(masks, step_size_samples)
    return (audio[0:len(vad_masks)], vad_masks, fs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='speech audio file (.wav)')
    parser.add_argument('--wavfile', required=True)
    args = parser.parse_args()

    audio, vad_masks, fs = speech_vad(args.wavfile)

    # plot vad masks
    xt = np.arange(0, len(audio), 1) / fs *1000
    plt.figure()
    plt.plot(xt, audio, color='blue')
    plt.xlabel('time (ms)')
    plt.ylabel('amplitude')
    plt.title('audio signal')
    plt.plot(xt, vad_masks*np.amax(audio), color="red")
    plt.show()

