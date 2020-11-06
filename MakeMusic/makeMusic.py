from scipy.io.wavfile import write,read
import numpy as np
import subprocess as sp
import shlex
import os
import matplotlib.pyplot as plot

vol = 0.5
fs = 44100
duration = 5
freq = np.array([440,523.25],ndmin=2)
# freq = np.reshape(1,2)

class audio:
    def __init__(self,sampling_frequency,duration_of_output_audio,volume,frequency_array):
        self.fs = sampling_frequency
        self.freq = frequency_array
        self.duration = duration_of_output_audio
        self.vol = volume

    def genwav(self):
        self.samples = (np.sin(2*np.pi*np.arange(self.fs*self.duration)*self.freq[0,0]/self.fs)).astype(np.float32)
        self.samples += (np.sin(2*np.pi*np.arange(self.fs*self.duration)*self.freq[0,1]/self.fs)).astype(np.float32)
        write("wavoutput.wav", self.fs, self.samples)

    def checkwav(self, checklen):
        if checklen == "full":
            n = self.fs*self.duration
        else:
            n = int(input("Number of samples: "))
        plot.plot(self.samples[0:n])
        plot.show()

    def playfile(self):
        filename = os.path.join("G:\\","Python","Music","wavoutput.wav")
        cmd = "ffplay {}".format(filename)
        sp.run(cmd, shell=True)

ad = audio(fs,duration,vol,freq)
ad.genwav()
ad.checkwav(None)
# ad.playfile()
