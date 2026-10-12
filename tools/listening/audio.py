# Sound, as a computer gets it: a list of numbers between −1 and 1, here 16,000 of them per second.
# A spectrogram turns that list into a picture: time runs from left to right, pitch from bottom to top.
import math
RATE, WIN, HOP, FRAMES, BANDS = 16000, 512, 500, 32, 32
# QUIET: a recording whose loudest point is quieter than this (in decibels) is measured from QUIET instead, so near-silence
# is not stretched to fill the whole range. Every recording here is louder.
QUIET = -20

# The Fourier transform of one short window says how strongly each frequency (pitch) is present in it.
# This is the fast version, the FFT: it reorders the samples, then combines pairs, then pairs of pairs.
def fft(re, im):
    n, j = len(re), 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            re[i], re[j] = re[j], re[i]
            im[i], im[j] = im[j], im[i]
    size = 2
    while size <= n:
        half, step = size // 2, -2 * math.pi / size
        for start in range(0, n, size):
            for k in range(half):
                c, s, a = math.cos(step * k), math.sin(step * k), start + k
                b = a + half
                tr, ti = re[b] * c - im[b] * s, re[b] * s + im[b] * c
                re[b], im[b] = re[a] - tr, im[a] - ti
                re[a], im[a] = re[a] + tr, im[a] + ti
        size *= 2

# Each window fades in and out (a Hann window), so its sudden edges add far fewer frequencies that are not there.
# We hear pitch roughly on a log scale, so the 32 bands are narrow at low frequencies and wide at high ones
# (the mel scale). Each band adds up the power of the frequencies under a triangle.
HANN = [0.5 - 0.5 * math.cos(2 * math.pi * n / WIN) for n in range(WIN)]
def to_mel(f): return 2595 * math.log10(1 + f / 700)
def from_mel(m): return 700 * (10 ** (m / 2595) - 1)
def triangle(b):
    top = to_mel(RATE / 2)
    lo, mid, hi = (from_mel(i * top / (BANDS + 1)) for i in (b, b + 1, b + 2))
    def w(f): return 0.0 if f <= lo or f >= hi else (f - lo) / (mid - lo) if f < mid else (hi - f) / (hi - mid)
    return [w(k * RATE / WIN) for k in range(WIN // 2 + 1)]
MEL = [triangle(b) for b in range(BANDS)]

# The spectrogram of one second: for each of 32 moments, window the sound, find the power of every frequency,
# add it up in the 32 bands, and take the logarithm in decibels (we hear loudness on a log scale too).
# Each value is then measured from the loudest point of the clip, so a quiet and a loud "seven" look alike:
# 0 dB below the loudest point becomes 1, 80 dB or more below becomes 0, in 16 levels. The top row is the highest band.
def spectrogram(samples):
    db = [0.0] * (BANDS * FRAMES)
    for t in range(FRAMES):
        re = [(samples[t * HOP + n] if t * HOP + n < len(samples) else 0.0) * HANN[n] for n in range(WIN)]
        im = [0.0] * WIN
        fft(re, im)
        for b in range(BANDS):
            power = sum(w * (re[k] * re[k] + im[k] * im[k]) for k, w in enumerate(MEL[b]) if w)
            db[(BANDS - 1 - b) * FRAMES + t] = 10 * math.log10(power + 1e-10)
    loudest = max(QUIET, *db)
    return [round(15 * max(0.0, 1 + (v - loudest) / 80)) / 15 for v in db]
