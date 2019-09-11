#!/usr/bin/python
"""
 SISO_RX.py

 Simple receiver. Receive signal and plot it continuously. Signal can be
 recorded and stored in HDF5 format. Alternatively, RX samples can be
 recorded into a binary file using the "write_to_file" function.

 Run both the SISO_TX.py script and this (SISO_RX.py) script
 at the same time.
 Multiple modes are supported:
   - BASIC: Receives and plots signal without recording it
   - REC: Receives and stores RX signal into file
   - REPLAY: Read back existing file and report RSSI

 AGC:
 To enable AGC, set the AGCen flag to 1. The AGC runs in the FPGA fabric.
 If enabled, LMS7 gains (LNA, TIA, PGA) start at their maximum values and
 get adjusted once AGC is triggered.
 To trigger AGC, first, both the AGC and packet detector need to be enabled via
 the following two commands:

    sdr.writeRegister("IRIS30", FPGA_IRIS030_WR_PKT_DET_ENABLE, 1)
    sdr.writeRegister("IRIS30", FPGA_IRIS030_WR_AGC_ENABLE_FLAG, 1)

 Second, we need a way to let the packet detector start looking at the
 incoming samples in order to determine whether a gain adjustment is needed
 To do so, we need to use the following commands:

    sdr.writeRegister("IRIS30", FPGA_IRIS030_WR_PKT_DET_NEW_FRAME, 1)

 Usage example: python3 SISO_RX.py --serial="RF3C000034" --rxMode="REC" --AGCen=1

---------------------------------------------------------------------
 Copyright © 2018-2019. Rice University.
 RENEW OPEN SOURCE LICENSE: http://renew-wireless.org/license
 ---------------------------------------------------------------------
"""

import sys
sys.path.append('../IrisUtils/')
sys.path.append('../IrisUtils/data_in/')

import SoapySDR
import numpy as np
import time
import datetime
import os
import math
import signal
import threading
import matplotlib
import matplotlib.pyplot as plt
import collections
import logging
import pdb
from SoapySDR import *              # SOAPY_SDR_ constants
from optparse import OptionParser
from matplotlib import animation
from data_recorder import *
from find_lts import *
from digital_rssi import *
from bandpower import *
from file_rdwr import *
from fft_power import *


#########################################
#            Global Parameters          #
#########################################
sdr = None
rxStream = None
recorder = None
FIG_LEN = 16384   
Rate = 5e6 
fft_size = 2**12 # 1024
numBufferSamps = 1000
rssiPwrBuffer = collections.deque(maxlen=numBufferSamps)
timePwrBuffer = collections.deque(maxlen=numBufferSamps)
freqPwrBuffer = collections.deque(maxlen=numBufferSamps)
noisPwrBuffer = collections.deque(maxlen=numBufferSamps)
rssiPwrBuffer_fpga = collections.deque(maxlen=numBufferSamps)
frameCounter = 0


########################################
#               LOGGER                 #
########################################
# SOAPY_SDR_FATAL    = 1, //!< A fatal error. The application will most likely terminate. This is the highest priority.
# SOAPY_SDR_CRITICAL = 2, //!< A critical error. The application might not be able to continue running successfully.
# SOAPY_SDR_ERROR    = 3, //!< Error.An operation didn't complete successfully, but application as a whole not affected.
# SOAPY_SDR_WARNING  = 4, //!< A warning. An operation completed with an unexpected result.
# SOAPY_SDR_NOTICE   = 5, //!< A notice, which is an information with just a higher priority.
# SOAPY_SDR_INFO     = 6, //!< An informational message, usually denoting the successful completion of an operation.
# SOAPY_SDR_DEBUG    = 7, //!< A debugging message.
# SOAPY_SDR_TRACE    = 8, //!< A tracing message. This is the lowest priority.
# SOAPY_SDR_SSI      = 9, //!< Streaming status indicators such as "U" (underflow) and "O" (overflow).
logLevel = 6         # 4:WARNING, 6:WARNING+INFO, 7:WARNING+INFO+DEBUG...
SoapySDR.SoapySDR_setLogLevel(logLevel)
logging.basicConfig(filename='./data_out/debug_SISO_RX.log',
                    level=logging.DEBUG,
                    format='[%(levelname)s] (%(threadName)-10s) %(asctime)s %(message)s',)


#########################################
#                Registers              #
#########################################
# Reset
RF_RST_REG = 48

# AGC registers Set
FPGA_IRIS030_WR_AGC_ENABLE_FLAG = 232
FPGA_IRIS030_WR_AGC_RESET_FLAG = 236
FPGA_IRIS030_WR_IQ_THRESH = 240
FPGA_IRIS030_WR_NUM_SAMPS_SAT = 244
FPGA_IRIS030_WR_MAX_NUM_SAMPS_AGC = 248
FPGA_IRIS030_WR_RSSI_TARGET = 252
FPGA_IRIS030_WR_WAIT_COUNT_THRESH = 256
FPGA_IRIS030_WR_AGC_SMALL_JUMP = 260
FPGA_IRIS030_WR_AGC_BIG_JUMP = 264
FPGA_IRIS030_WR_AGC_TEST_GAIN_SETTINGS = 268
FPGA_IRIS030_WR_AGC_LNA_IN = 272
FPGA_IRIS030_WR_AGC_TIA_IN = 276
FPGA_IRIS030_WR_AGC_PGA_IN = 280

# RSSI register Set
FPGA_IRIS030_RD_MEASURED_RSSI = 284

# Packet Detect Register Set
FPGA_IRIS030_WR_PKT_DET_THRESH = 288
FPGA_IRIS030_WR_PKT_DET_NUM_SAMPS = 292
FPGA_IRIS030_WR_PKT_DET_ENABLE = 296
FPGA_IRIS030_WR_PKT_DET_NEW_FRAME = 300


#########################################
#             Create Plots              #
#########################################
matplotlib.rcParams.update({'font.size': 10})
fig = plt.figure(figsize=(20, 8), dpi=120)
fig.subplots_adjust(hspace=.5, top=.85)

ax1 = fig.add_subplot(6, 1, 1)
ax1.grid(True)
ax1.set_title('Waveform capture')
title = ax1.text(0.5, 1, '|', ha="center")
ax1.set_ylabel('Signal')
ax1.set_xlabel('Sample index')
line1, = ax1.plot([], [], label='ChA I', animated=True)
line2, = ax1.plot([], [], label='ChB I', animated=True)
ax1.set_ylim(-1, 1)
ax1.set_xlim(0, FIG_LEN)
ax1.legend(fontsize=10)

ax2 = fig.add_subplot(6, 1, 2)
ax2.grid(True)
ax2.set_xlabel('Sample index')
ax2.set_ylabel('Amplitude')
line3, = ax2.plot([], [], label='ChA', animated=True)
line4, = ax2.plot([], [], label='ChB', animated=True)
ax2.set_ylim(-2, 2)
ax2.set_xlim(0, FIG_LEN)
ax2.legend(fontsize=10)

ax3 = fig.add_subplot(6, 1, 3)
ax3.grid(True)
ax3.set_xlabel('Sample index')
ax3.set_ylabel('Phase')
line5, = ax3.plot([], [], label='Phase ChA', animated=True)
line6, = ax3.plot([], [], label='Phase ChB', animated=True)
line7, = ax3.plot([], [], label='Delta A-B', animated=True)
ax3.set_ylim(-3.2, 3.2)
ax3.set_xlim(0, FIG_LEN)
ax3.legend(fontsize=10)

ax4 = fig.add_subplot(6, 1, 4)
ax4.grid(True)
ax4.set_xlabel('Frequency (Hz)')
ax4.set_ylabel('Power (dB)')
line8, = ax4.plot([], [], label='FFT ChA', animated=True)
ax4.set_ylim(-110, 0)
freqScale = np.arange(-Rate / 2, Rate / 2, Rate / fft_size)
#freqScale = np.arange(-Rate / 2 / 1e6, Rate / 2 / 1e6, Rate / fft_size / 1e6)[:fft_size]
ax4.set_xlim(freqScale[0], freqScale[-1])
ax4.legend(fontsize=10)

ax5 = fig.add_subplot(6, 1, 5)
ax5.grid(True)
ax5.set_xlabel('Real-Time Samples')
ax5.set_ylabel('Power (dB)')
line9, = ax5.plot([], [], label='Digital RSSI', animated=True)
line10, = ax5.plot([], [], label='TimeDomain Sig Pwr', animated=True)
line11, = ax5.plot([], [], label='FreqDomain Sig Pwr', animated=True, linestyle='dashed')
line12, = ax5.plot([], [], label='Noise Floor', animated=True)
line13, = ax5.plot([], [], label='RSSI_FPGA_Pwr', animated=True, linestyle='dashed')
ax5.set_ylim(-100, 10)
ax5.set_xlim(0, numBufferSamps * 1.5)
ax5.legend(fontsize=10)

ax6 = fig.add_subplot(6, 1, 6)
ax6.grid(True)
ax6.set_xlabel('Sample index')
ax6.set_ylabel('Correlation Peaks')
line14, = ax6.plot([], [], label='Corr I ChA', animated=True)
ax6.set_ylim(-10, 50)
ax6.set_xlim(0, 2**12)
ax6.legend(fontsize=10)


#########################################
#              Functions                #
#########################################
def init():
    """ Initialize plotting objects """
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    line4.set_data([], [])
    line5.set_data([], [])
    line6.set_data([], [])
    line7.set_data([], [])
    line8.set_data([], [])
    line9.set_data([], [])
    line10.set_data([], [])
    line11.set_data([], [])
    line12.set_data([], [])
    line13.set_data([], [])
    line14.set_data([], [])
    return line1, line2, line3, line4, line5, line6, line7, line8, line9, line10, line11, line12, line13, line14


def register_setup():
    # AGC setup
    sdr.writeRegister("IRIS30", FPGA_IRIS030_WR_AGC_ENABLE_FLAG, 0)         # Enable AGC Flag (set to 0 initially)
    sdr.writeRegister("IRIS30", FPGA_IRIS030_WR_AGC_RESET_FLAG, 1)          # Reset AGC Flag
    sdr.writeRegister("IRIS30", FPGA_IRIS030_WR_IQ_THRESH, 10300)           # Saturation Threshold: 10300 about -6dBm
    sdr.writeRegister("IRIS30", FPGA_IRIS030_WR_NUM_SAMPS_SAT, 3)           # Number of samples needed to claim sat.
    sdr.writeRegister("IRIS30", FPGA_IRIS030_WR_MAX_NUM_SAMPS_AGC, 20)      # Threshold at which AGC stops
    sdr.writeRegister("IRIS30", FPGA_IRIS030_WR_WAIT_COUNT_THRESH, 160)     # Gain settle takes about 20 samps(value=20)
    sdr.writeRegister("IRIS30", FPGA_IRIS030_WR_AGC_RESET_FLAG, 0)          # Clear AGC reset flag
    sdr.writeRegister("IRIS30", FPGA_IRIS030_WR_AGC_BIG_JUMP, 15)           # Drop gain at initial saturation detection
    sdr.writeRegister("IRIS30", FPGA_IRIS030_WR_AGC_SMALL_JUMP, 5)          # Drop gain at subsequent sat. detections
    sdr.writeRegister("IRIS30", FPGA_IRIS030_WR_RSSI_TARGET, 20)            # RSSI Target for AGC: ideally around 14
    sdr.writeRegister("IRIS30", FPGA_IRIS030_WR_AGC_TEST_GAIN_SETTINGS, 0)  # Enable only for testing gain settings

    # Pkt Detect Setup
    sdr.writeRegister("IRIS30", FPGA_IRIS030_WR_PKT_DET_THRESH, 0)          # RSSI value at which Pkt is detected
    sdr.writeRegister("IRIS30", FPGA_IRIS030_WR_PKT_DET_NUM_SAMPS, 5)       # Number of samples needed to detect frame
    sdr.writeRegister("IRIS30", FPGA_IRIS030_WR_PKT_DET_ENABLE, 0)          # Enable packet detection flag
    sdr.writeRegister("IRIS30", FPGA_IRIS030_WR_PKT_DET_NEW_FRAME, 0)       # Finished last frame? (set to 0 initially)


def rxsamples_app(srl, freq, gain, num_samps, recorder, agc_en, wait_trigger):
    """
    Initialize IRIS parameters and animation kick-off
    """

    # Global declarations
    global sdr, rxStream, freqScale, Rate

    # Instantiate device
    sdr = SoapySDR.Device(dict(serial=srl))
    info = sdr.getHardwareInfo()
    print(info)

    # Set params on both channels (both RF chains)
    for ch in [0, 1]:
        #sdr.setBandwidth(SOAPY_SDR_RX, ch, 3*Rate)
        #sdr.setBandwidth(SOAPY_SDR_TX, ch, 3*Rate)
        sdr.setFrequency(SOAPY_SDR_RX, ch, freq)
        sdr.setSampleRate(SOAPY_SDR_RX, ch, Rate)
        sdr.setFrequency(SOAPY_SDR_TX, ch, freq)
        sdr.setSampleRate(SOAPY_SDR_TX, ch, Rate)
        if "CBRS" in info["frontend"]:
            sdr.setGain(SOAPY_SDR_RX, ch, 'LNA2', gain[5])  # [0,17]
            sdr.setGain(SOAPY_SDR_RX, ch, 'LNA1', gain[4])  # [0,33]
            sdr.setGain(SOAPY_SDR_RX, ch, 'ATTN', gain[3])  # [-18,0]
        sdr.setGain(SOAPY_SDR_RX, ch, 'LNA', gain[2])       # [0,30]
        sdr.setGain(SOAPY_SDR_RX, ch, 'TIA', gain[1])       # [0,12]
        sdr.setGain(SOAPY_SDR_RX, ch, 'PGA', gain[0])       # [-12,19]
        # sdr.setAntenna(SOAPY_SDR_RX, ch, "TRX")
        sdr.setDCOffsetMode(SOAPY_SDR_RX, ch, True)

    print("Number of Samples %d " % num_samps)
    print("Frequency has been set to %f" % sdr.getFrequency(SOAPY_SDR_RX, 0))
    sdr.writeRegister("RFCORE", 120, 0)

    # Setup RX stream
    rxStream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0, 1])

    # Initialize registers to be used by AGC and packet detect FPGA cores
    register_setup()
    # RSSI read setup
    setUpDigitalRssiMode(sdr)

    anim = animation.FuncAnimation(fig, animate, init_func=init, fargs=(num_samps, recorder, agc_en, wait_trigger), frames=100,
                                   interval=100, blit=True)
    plt.show()


def animate(i, num_samps, recorder, agc_en, wait_trigger):
    global sdr, rxStream, freqScale, sampsRx, frameCounter, fft_size, Rate

    # Trigger AGC
    if agc_en:
        if frameCounter == 10:
            print(" *** ENABLE AGC/PKT DETECT *** ")
            sdr.writeRegister("IRIS30", FPGA_IRIS030_WR_PKT_DET_ENABLE, 1)
            sdr.writeRegister("IRIS30", FPGA_IRIS030_WR_AGC_ENABLE_FLAG, 1)
        if frameCounter == 20:
            print(" *** DONE WITH PREVIOUS FRAME, ASSUME NEW FRAME INCOMING *** ")
            sdr.writeRegister("IRIS30", FPGA_IRIS030_WR_PKT_DET_NEW_FRAME, 1)
            sdr.writeRegister("IRIS30", FPGA_IRIS030_WR_PKT_DET_NEW_FRAME, 0)  # DO NOT REMOVE ME!! Disable right away

    # Count number of frames received
    frameCounter = frameCounter + 1

    # Read samples into this buffer
    sampsRx = [np.zeros(num_samps, np.complex64), np.zeros(num_samps, np.complex64)]
    buff0 = sampsRx[0]  # RF Chain 1
    buff1 = sampsRx[1]  # RF Chain 2

    flags = SOAPY_SDR_END_BURST
    if wait_trigger: flags |= SOAPY_SDR_WAIT_TRIGGER
    sdr.activateStream(rxStream,
        flags,    # flags
        0,                      # timeNs (dont care unless using SOAPY_SDR_HAS_TIME)
        buff0.size)             # numElems - this is the burst size
    sr = sdr.readStream(rxStream, [buff0, buff1], buff0.size)
    if sr.ret != buff0.size:
        print("Read RX burst of %d, requested %d" % (sr.ret, buff0.size))

    # DC removal
    for i in [0, 1]:
        sampsRx[i] -= np.mean(sampsRx[i])

    # Find LTS peaks (in case LTSs were sent)
    lts_thresh = 0.8
    a, b, peaks0 = find_lts(sampsRx[0], thresh=lts_thresh)
    a, b, peaks1 = find_lts(sampsRx[1], thresh=lts_thresh)

    # If recording samples
    if recorder is not None: 
        frame = np.empty((2, buff0.size), dtype='complex64')
        frame[0] = sampsRx[0]
        frame[1] = sampsRx[1]
        recorder.save_frame(sampsRx, sr.timeNs)

    # Store received samples in binary file (second method of storage)
    write_to_file('./data_out/rxsamps', sampsRx)

    # Magnitude of IQ Samples (RX RF chain A)
    I = np.real(sampsRx[0])
    Q = np.imag(sampsRx[0])
    IQmag = np.mean(np.sqrt(I**2 + Q**2))

    # Retrieve RSSI measured from digital samples at the LMS7, and convert to PWR in dBm
    agc_avg = 7
    rssi, PWRdBm = getDigitalRSSI(sdr, agc_avg)
    PWRdB = PWRdBm - 30

    # Compute Power of Time Domain Signal
    sigRms = np.sqrt(np.mean(sampsRx[0] * np.conj(sampsRx[0])))
    sigPwr = np.real(sigRms) ** 2
    sigPwr_dB = 10 * np.log10(sigPwr)
    sigPwr_dBm = 10 * np.log10(sigPwr / 1e-3)

    # Compute Power of Frequency Domain Signal (FFT)
    f1, powerBins, noiseFloor, pks = fft_power(sampsRx[0], Rate, num_bins=fft_size, peak=1.0,
                                               scaling='spectrum', peak_thresh=20)
    fftPower = bandpower(sampsRx[0], Rate, 0, Rate / 2)
    fftPower_dB = 10 * np.log10(fftPower)
    fftPower_dBm = 10 * np.log10(fftPower / 1e-3)

    # Retrieve RSSI computed in the FPGA
    rssi_fpga = int(sdr.readRegister("IRIS30", FPGA_IRIS030_RD_MEASURED_RSSI))
    Vrms_fpga = (rssi_fpga / 2.0 ** 16) * (1 / np.sqrt(2.0))  # Vrms = Vpeak/sqrt(2) (In Volts) - 16 bit value
    PWRrms_fpga = (Vrms_fpga ** 2.0) / 50.0                   # 50 Ohms load (PWRrms in Watts)
    PWRdBm_fpga = 10.0 * np.log10(PWRrms_fpga) + 30           # P(dBm)=10*log10(Prms/1mW)  OR  P(dBm)=10*log10(Prms)+30

    # Circular buffer - continuously display data
    rssiPwrBuffer.append(PWRdBm)
    timePwrBuffer.append(sigPwr_dB)
    freqPwrBuffer.append(fftPower_dB)
    noisPwrBuffer.append(noiseFloor)
    rssiPwrBuffer_fpga.append(PWRdBm_fpga)

    lna_rd = sdr.getGain(SOAPY_SDR_RX, 0, 'LNA')    # ChanA (0)
    tia_rd = sdr.getGain(SOAPY_SDR_RX, 0, 'TIA')    # ChanA (0)
    pga_rd = sdr.getGain(SOAPY_SDR_RX, 0, 'PGA')    # ChanA (0)
    print("RSSI: {} \t PWR: {} \t LNA: {} \t TIA: {} \t PGA: {}".format(rssi_fpga, PWRdBm_fpga, lna_rd, tia_rd, pga_rd))

    # Fill out data structures with measured data
    line1.set_data(range(buff0.size), np.real(sampsRx[0]))
    line2.set_data(range(buff0.size), np.imag(sampsRx[0]))
    line3.set_data(range(buff0.size), np.abs(sampsRx[0]))
    line4.set_data(range(buff0.size), np.abs(sampsRx[1]))
    line5.set_data(range(buff0.size), np.angle(sampsRx[0]))
    line6.set_data(range(buff0.size), np.angle(sampsRx[1]))
    line7.set_data(range(buff0.size), np.angle(sampsRx[0] * np.conj(sampsRx[1])))
    line8.set_data(f1, powerBins)
    line9.set_data(range(len(rssiPwrBuffer)), rssiPwrBuffer)
    line10.set_data(range(len(timePwrBuffer)), timePwrBuffer)
    line11.set_data(range(len(freqPwrBuffer)), freqPwrBuffer)
    line12.set_data(range(len(noisPwrBuffer)), noisPwrBuffer)
    line13.set_data(range(len(rssiPwrBuffer_fpga)), rssiPwrBuffer_fpga)
    line14.set_data(range(buff0.size), np.real(peaks0[:buff0.size]))

    return line1, line2, line3, line4, line5, line6, line7, line8, line9, line10, line11, line12, line13, line14


def replay(name, leng):
    """
    Open existing file (where samples were previously recorded) and analyze
    """

    print("File name %s, batch length %s" % (name, str(leng)))
    h5file = h5py.File('./data_out/'+name, 'r')
    samples = h5file['Samples']
    leng = h5file.attrs['frame_length']
    numFrame = samples.shape[0]
    avg_rssi = [0, 0]
    rssi = [0, 0]
    for i in range(numFrame):
        sampsRx = samples[i, :, :]
        rssi[0] = np.mean(np.power(np.abs(sampsRx[0]),2)) 
        rssi[1] = np.mean(np.power(np.abs(sampsRx[1]),2)) 
        log_rssi = 10*np.log10(rssi)
        avg_rssi[0] += rssi[0]
        avg_rssi[1] += rssi[1]
        print("Measured RSSI from batch %d: (%f, %f)" % (i, log_rssi[0], log_rssi[1]))
    avg_rssi = 10*np.log10([x/numFrame for x in avg_rssi])
    print("Average Measured RSSI: (%f, %f)" % (avg_rssi[0], avg_rssi[1]))


#########################################
#                  Main                 #
#########################################
def main():
    parser = OptionParser()
    parser.add_option("--label", type="string", dest="label", help="label for recorded file name", default="rx2.600GHz_TEST.hdf5")
    parser.add_option("--lna", type="float", dest="lna", help="Lime Chip Rx LNA gain [0:30](dB)", default=14.0)
    parser.add_option("--tia", type="float", dest="tia", help="Lime Chip Rx TIA gain [0,3,9,12] (dB)", default=0.0)
    parser.add_option("--pga", type="float", dest="pga", help="Lime Chip Rx PGA gain [-12:19] (dB)", default=0.0)
    parser.add_option("--lna1", type="float", dest="lna1", help="BRS/CBRS Front-end LNA1 gain stage [0:33] (dB)", default=33.0)
    parser.add_option("--lna2", type="float", dest="lna2", help="BRS/CBRS Front-end LNA2 gain [0:17] (dB)", default=17.0)
    parser.add_option("--attn", type="float", dest="rxattn", help="BRS/CBRS Front-end ATTN gain stage [-18:6:0] (dB)", default=0.0)
    parser.add_option("--latitude", type="float", dest="latitude", help="Latitude", default=0.0)
    parser.add_option("--longitude", type="float", dest="longitude", help="Longitude", default=0.0)
    parser.add_option("--elevation", type="float", dest="elevation", help="Elevation", default=0.0)
    parser.add_option("--freq", type="float", dest="freq", help="Optional Rx freq (Hz)", default=2.5e9)
    parser.add_option("--numSamps", type="int", dest="numSamps", help="Num samples to receive", default=16384)
    parser.add_option("--serial", type="string", dest="serial", help="Serial number of the device", default="")
    parser.add_option("--rxMode", type="string", dest="rxMode", help="RX Mode, Options:BASIC/REC/REPLAY", default="BASIC")
    parser.add_option("--AGCen", type="int", dest="AGCen", help="Enable AGC Flag. Options:0/1", default=0)
    parser.add_option("--wait-trigger", action="store_true", dest="wait_trigger", help="wait for a trigger to start a frame",default=False)
    (options, args) = parser.parse_args()

    # Verify RX Mode
    if not (options.rxMode == "BASIC" or options.rxMode == "REC" or options.rxMode == "REPLAY"):
        raise AssertionError("Invalid RX Mode")

    # Current time
    now = datetime.datetime.now()
    print(now)

    # Display parameters
    print("\n")
    print("========== RX PARAMETERS =========")
    print("Receiving signal on board {}".format(options.serial))
    print("Sample Rate (sps): {}".format(Rate))
    print("Lime Rx Gain (dB) - LNA: {}, TIA: {}, PGA: {}".format(options.lna, options.tia, options.pga))
    print("BRS/CBRS Rx Gain (dB) - LNA1: {}, LNA2: {}, ATTN: {}".format(options.lna1, options.lna2, options.rxattn))
    print("Frequency (Hz): {}".format(options.freq))
    print("RX Mode: {}".format(options.rxMode))
    print("Number of Samples: {}".format(options.numSamps))
    if options.AGCen: print("** AGC ENABLED **")
    print("==================================")
    print("\n")

    # Set gains to max. if AGC enabled.
    if options.AGCen:
        options.lna = 30
        options.tia = 12
        options.pga = 19

    # If recording file
    recorder = None
    if options.rxMode == "REC":
        filename = "./data_out/rx" + '%1.3f' % (float(options.freq)/1e9) + 'GHz_' + options.label + '.hdf5'
        recorder = DataRecorder(options.label,
                                options.serial,
                                options.freq,
                                options.lna,
                                options.tia,
                                options.pga,
                                options.lna1,
                                options.lna2,
                                options.rxattn,
                                options.numSamps,
                                options.latitude,
                                options.longitude,
                                options.elevation)
        recorder.init_h5file(filename=filename)

    # IF REPLAY
    if options.rxMode == "REPLAY":
        replay(options.label, int(options.numSamps))

    else:
        rxsamples_app(
            srl=options.serial,
            freq=options.freq,
            gain=[options.pga, options.tia, options.lna, options.rxattn, options.lna1, options.lna2],
            num_samps=options.numSamps,
            recorder=recorder,
            agc_en=options.AGCen,
            wait_trigger=options.wait_trigger
        )


if __name__ == '__main__': 
    main()
