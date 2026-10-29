"""PCM preparation and a persistent Windows waveform output device."""
from dataclasses import dataclass
import ctypes as ct
import os
from pathlib import Path
import wave
import numpy as np

RATE = 24000

@dataclass(frozen=True)
class Audio:
    pcm: bytes
    @property
    def duration(self):
        return len(self.pcm) / (RATE * 2)

def read_audio(path):
    with wave.open(str(path), "rb") as wav:
        channels, width, rate = wav.getnchannels(), wav.getsampwidth(), wav.getframerate()
        if width != 2 or wav.getcomptype() != "NONE":
            raise ValueError("Clip cần WAV PCM 16-bit.")
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32)
    samples = samples.reshape(-1, channels).mean(axis=1)
    # Keep a quiet margin so weak consonants are not clipped.
    nonquiet = np.flatnonzero(np.abs(samples) > max(64, float(np.max(np.abs(samples), initial=0)) * .003))
    if len(nonquiet):
        left = max(0, int(nonquiet[0])-round(rate*.02))
        right = min(len(samples), int(nonquiet[-1])+round(rate*.04)+1)
        samples = samples[left:right]
    if rate != RATE and len(samples):
        count = max(1, round(len(samples)*RATE/rate))
        samples = np.interp(np.arange(count)*rate/RATE, np.arange(len(samples)), samples)
    return Audio(np.clip(samples, -32768, 32767).astype("<i2").tobytes())

def join_audio(parts, gap_ms=25):
    gap = bytes(round(RATE*gap_ms/1000)*2)
    return Audio(gap.join(part.pcm for part in parts))

def save_audio(audio, path):
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(RATE)
        wav.writeframes(audio.pcm)

class WaveOutput:
    """One waveOut device for the worker lifetime. Buffers stay alive until unprepared."""
    name = "Windows PCM persistent"
    def __init__(self):
        if os.name != "nt":
            raise RuntimeError("WaveOutput requires Windows.")
        class Format(ct.Structure):
            _pack_ = 1
            _fields_ = [("tag",ct.c_uint16),("channels",ct.c_uint16),("rate",ct.c_uint32),
                        ("bytes_sec",ct.c_uint32),("align",ct.c_uint16),("bits",ct.c_uint16),("extra",ct.c_uint16)]
        class Header(ct.Structure):
            _fields_ = [("data",ct.c_void_p),("length",ct.c_uint32),("recorded",ct.c_uint32),
                        ("user",ct.c_size_t),("flags",ct.c_uint32),("loops",ct.c_uint32),
                        ("next",ct.c_void_p),("reserved",ct.c_size_t)]
        self.Header = Header
        self.api = ct.WinDLL("winmm")
        self.handle = ct.c_void_p()
        self.header = self.buffer = None
        self.api.waveOutOpen.argtypes = [ct.POINTER(ct.c_void_p),ct.c_uint32,ct.POINTER(Format),
                                         ct.c_size_t,ct.c_size_t,ct.c_uint32]
        for name in ("waveOutPrepareHeader","waveOutUnprepareHeader","waveOutWrite"):
            getattr(self.api,name).argtypes = [ct.c_void_p,ct.POINTER(Header),ct.c_uint32]
        for name in ("waveOutReset","waveOutClose"):
            getattr(self.api,name).argtypes = [ct.c_void_p]
        for name in ("waveOutOpen","waveOutPrepareHeader","waveOutUnprepareHeader","waveOutWrite","waveOutReset","waveOutClose"):
            getattr(self.api,name).restype = ct.c_uint32
        fmt = Format(1, 1, RATE, RATE*2, 2, 16, 0)
        self._check(self.api.waveOutOpen(ct.byref(self.handle),0xffffffff,ct.byref(fmt),0,0,0))

    @staticmethod
    def _check(code):
        if code:
            raise OSError(f"Windows audio error {code}")

    def _release(self):
        if self.header is not None:
            self._check(self.api.waveOutUnprepareHeader(self.handle, ct.byref(self.header), ct.sizeof(self.Header)))
            self.header = self.buffer = None

    def play(self, audio):
        self.stop()
        self.buffer = ct.create_string_buffer(audio.pcm)
        self.header = self.Header(ct.cast(self.buffer,ct.c_void_p),len(audio.pcm),0,0,0,0,None,0)
        self._check(self.api.waveOutPrepareHeader(self.handle,ct.byref(self.header),ct.sizeof(self.Header)))
        self._check(self.api.waveOutWrite(self.handle,ct.byref(self.header),ct.sizeof(self.Header)))

    def busy(self):
        return self.header is not None and not bool(self.header.flags & 1)  # WHDR_DONE

    def stop(self):
        if self.header is not None:
            if self.busy():
                self._check(self.api.waveOutReset(self.handle))
            self._release()

    def close(self):
        if self.handle:
            self.stop()
            self._check(self.api.waveOutClose(self.handle))
            self.handle = None

class PortableOutput:
    name = "sounddevice"
    def __init__(self):
        import sounddevice
        self.sd = sounddevice
    def play(self,audio):
        self.sd.play(np.frombuffer(audio.pcm,dtype="<i2"), RATE)
    def busy(self):
        return self.sd.get_stream().active
    def stop(self):
        self.sd.stop()
    def close(self):
        self.stop()

def create_output():
    return WaveOutput() if os.name == "nt" else PortableOutput()
