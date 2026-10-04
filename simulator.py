"""
simulator.py  --  Signal generation for the AI Wireless Signal Analyzer dataset.

Mirrors the signal chain used by GNU Radio / RadioML dataset generation
(O'Shea & West, 2016):  source -> modulator -> pulse shaping -> channel -> AWGN.
Everything is pure NumPy/SciPy, so it runs anywhere with no SDR hardware.

Every generator returns a complex baseband frame of exactly FRAME_LEN samples.
"""
import numpy as np
from scipy.signal import firwin, hilbert, lfilter

FRAME_LEN = 128      # complex samples per frame (same as RadioML 2016.10A)
SPS = 8              # samples per symbol for digital modulations
ROLLOFF = 0.35       # root-raised-cosine roll-off
_PAD = 256           # extra samples generated, then cropped (removes filter transients)

# ----------------------------------------------------------------------------
# Class definitions
# ----------------------------------------------------------------------------
MODULATIONS = ["BPSK", "QPSK", "8PSK", "QAM16", "QAM64", "PAM4",
               "GFSK", "CPFSK", "WBFM", "AM-DSB", "AM-SSB"]
MOD_ID = {m: i for i, m in enumerate(MODULATIONS)}

ANOMALY_TYPES = ["none", "jamming_tone", "clipping", "impulsive_noise",
                 "unknown_ofdm", "unknown_chirp"]
ANOM_ID = {a: i for i, a in enumerate(ANOMALY_TYPES)}

CHANNELS = ["awgn", "fading"]
CHAN_ID = {c: i for i, c in enumerate(CHANNELS)}

SNR_LIST = list(range(-20, 20, 2))   # -20 ... +18 dB, 20 levels (same grid as RadioML)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def rrc_taps(sps=SPS, beta=ROLLOFF, span=8):
    """Root-raised-cosine pulse-shaping filter."""
    n = np.arange(-span * sps // 2, span * sps // 2 + 1) / sps
    taps = np.zeros_like(n)
    for i, t in enumerate(n):
        if abs(t) < 1e-9:
            taps[i] = 1.0 - beta + 4 * beta / np.pi
        elif abs(abs(4 * beta * t) - 1.0) < 1e-9:
            taps[i] = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            num = np.sin(np.pi * t * (1 - beta)) + 4 * beta * t * np.cos(np.pi * t * (1 + beta))
            den = np.pi * t * (1 - (4 * beta * t) ** 2)
            taps[i] = num / den
    return taps / np.sqrt(np.sum(taps ** 2))

_RRC = rrc_taps()


def _n_symbols():
    return (FRAME_LEN + 2 * _PAD) // SPS + 1


def _shape(symbols):
    """Upsample complex symbols by SPS and apply RRC pulse shaping."""
    up = np.zeros(len(symbols) * SPS, dtype=complex)
    up[::SPS] = symbols
    return np.convolve(up, _RRC, mode="same")


def _crop(x, rng):
    """Random crop of FRAME_LEN samples (gives random symbol-timing offset)."""
    start = rng.integers(_PAD // 2, len(x) - FRAME_LEN - _PAD // 2)
    return x[start:start + FRAME_LEN]


def _message(rng, n):
    """Band-limited random 'audio' message in [-1, 1] (stand-in for speech/music)."""
    white = rng.standard_normal(n + 400)
    h = firwin(101, 0.12)                     # cut-off = 0.12 * Nyquist
    m = lfilter(h, 1.0, white)[200:200 + n]
    return m / (np.max(np.abs(m)) + 1e-12)


# ----------------------------------------------------------------------------
# Digital linear modulations
# ----------------------------------------------------------------------------
def _psk(rng, M):
    k = rng.integers(0, M, _n_symbols())
    return _shape(np.exp(1j * (2 * np.pi * k / M + (np.pi / M if M == 4 else 0))))


def gen_bpsk(rng): return _crop(_shape(rng.choice([-1.0, 1.0], _n_symbols()).astype(complex)), rng)
def gen_qpsk(rng): return _crop(_psk(rng, 4), rng)
def gen_8psk(rng): return _crop(_psk(rng, 8), rng)


def _qam(rng, M):
    side = int(np.sqrt(M))
    levels = np.arange(-(side - 1), side, 2)
    sym = rng.choice(levels, _n_symbols()) + 1j * rng.choice(levels, _n_symbols())
    return _shape(sym)


def gen_qam16(rng): return _crop(_qam(rng, 16), rng)
def gen_qam64(rng): return _crop(_qam(rng, 64), rng)


def gen_pam4(rng):
    sym = rng.choice([-3, -1, 1, 3], _n_symbols()).astype(complex)
    return _crop(_shape(sym), rng)


# ----------------------------------------------------------------------------
# Continuous-phase / frequency modulations
# ----------------------------------------------------------------------------
def _gauss_taps(bt=0.35, span=4):
    t = np.arange(-span * SPS // 2, span * SPS // 2 + 1) / SPS
    sigma = np.sqrt(np.log(2)) / (2 * np.pi * bt)
    g = np.exp(-t ** 2 / (2 * sigma ** 2))
    return g / g.sum()

_GAUSS = _gauss_taps()


def _cpm(rng, h, gaussian):
    bits = rng.choice([-1.0, 1.0], _n_symbols())
    up = np.repeat(bits, SPS)
    if gaussian:
        up = np.convolve(up, _GAUSS, mode="same")
    phase = np.pi * h * np.cumsum(up) / SPS               # h = modulation index
    return np.exp(1j * phase)


def gen_gfsk(rng):  return _crop(_cpm(rng, 0.5, True), rng)    # Gaussian-filtered FSK
def gen_cpfsk(rng): return _crop(_cpm(rng, 1.0, False), rng)   # rectangular CPFSK


# ----------------------------------------------------------------------------
# Analog modulations
# ----------------------------------------------------------------------------
def gen_wbfm(rng):
    m = _message(rng, FRAME_LEN + 2 * _PAD)
    phase = 2 * np.pi * 0.2 * np.cumsum(m)               # peak deviation = 0.2 * fs
    return _crop(np.exp(1j * phase), rng)


def gen_am_dsb(rng):
    m = _message(rng, FRAME_LEN + 2 * _PAD)
    return _crop((1.0 + 0.8 * m).astype(complex), rng)


def gen_am_ssb(rng):
    m = _message(rng, FRAME_LEN + 2 * _PAD)
    return _crop(hilbert(m), rng)                        # upper side-band analytic signal


GENERATORS = {
    "BPSK": gen_bpsk, "QPSK": gen_qpsk, "8PSK": gen_8psk, "QAM16": gen_qam16,
    "QAM64": gen_qam64, "PAM4": gen_pam4, "GFSK": gen_gfsk, "CPFSK": gen_cpfsk,
    "WBFM": gen_wbfm, "AM-DSB": gen_am_dsb, "AM-SSB": gen_am_ssb,
}


# ----------------------------------------------------------------------------
# Unknown / out-of-distribution signals (used as anomalies)
# ----------------------------------------------------------------------------
def gen_ofdm(rng, n_sc=16):
    """Simple OFDM-like multicarrier signal (QPSK on 16 sub-carriers, no cyclic prefix)."""
    n_blocks = (FRAME_LEN + 2 * _PAD) // n_sc + 2
    sym = np.exp(1j * (np.pi / 2 * rng.integers(0, 4, (n_blocks, n_sc)) + np.pi / 4))
    x = np.fft.ifft(sym, axis=1).reshape(-1) * np.sqrt(n_sc)
    start = rng.integers(0, len(x) - FRAME_LEN)
    return x[start:start + FRAME_LEN]


def gen_chirp(rng):
    """Linear FM chirp (radar-like sweep)."""
    t = np.arange(FRAME_LEN)
    f0 = rng.uniform(-0.4, 0.0)
    f1 = rng.uniform(0.0, 0.4)
    k = (f1 - f0) / FRAME_LEN
    return np.exp(2j * np.pi * (f0 * t + 0.5 * k * t ** 2))


# ----------------------------------------------------------------------------
# Channel
# ----------------------------------------------------------------------------
def _unit_power(x):
    return x / (np.sqrt(np.mean(np.abs(x) ** 2)) + 1e-12)


def apply_channel(x, channel, rng):
    """
    awgn   : random carrier-frequency offset + phase offset only.
    fading : plus 3-tap Rayleigh multipath with exponential power-delay profile.
    Output has unit average power, so SNR is well defined afterwards.
    """
    t = np.arange(len(x))
    cfo = rng.uniform(-0.01, 0.01)                       # +-1 % of sample-rate
    x = x * np.exp(1j * (2 * np.pi * cfo * t + rng.uniform(0, 2 * np.pi)))
    if channel == "fading":
        pdp = np.exp(-np.arange(3) / 1.0)
        pdp /= pdp.sum()
        taps = np.sqrt(pdp / 2) * (rng.standard_normal(3) + 1j * rng.standard_normal(3))
        x = np.convolve(x, taps, mode="same")
    return _unit_power(x)


def add_awgn(x, snr_db, rng):
    """Add complex white Gaussian noise for a given SNR (signal has unit power)."""
    sigma2 = 10 ** (-snr_db / 10)
    noise = np.sqrt(sigma2 / 2) * (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x)))
    return x + noise


# ----------------------------------------------------------------------------
# Anomaly injection (applied on top of a normal modulated frame)
# ----------------------------------------------------------------------------
def inject_jamming_tone(x, rng):
    """Strong continuous-wave tone; jammer-to-signal ratio 0 .. 10 dB."""
    jsr = 10 ** (rng.uniform(0, 10) / 10)
    f = rng.uniform(-0.45, 0.45)
    tone = np.sqrt(jsr) * np.exp(1j * (2 * np.pi * f * np.arange(len(x)) + rng.uniform(0, 2 * np.pi)))
    return x + tone


def inject_clipping(x, rng):
    """Amplifier saturation: hard-clip I and Q at a fraction of the peak."""
    lim = rng.uniform(0.25, 0.6) * max(np.max(np.abs(x.real)), np.max(np.abs(x.imag)))
    return np.clip(x.real, -lim, lim) + 1j * np.clip(x.imag, -lim, lim)


def inject_impulses(x, rng):
    """Sparse, very strong impulsive noise bursts (Bernoulli-Gaussian)."""
    mask = rng.random(len(x)) < rng.uniform(0.03, 0.10)
    amp = rng.uniform(4, 10)
    imp = amp * (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x))) * mask
    return x + imp
