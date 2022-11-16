import numpy as np
import esig
import matplotlib.pyplot as plt

from numpy.random import default_rng
rng = default_rng(12345)

upper_limit = 2*np.pi
depth = 2
noise_variance = 0.1 

def make_noisy(signal):
    return signal + rng.normal(0.0, noise_variance, size=signal.shape)


def signal_a(count):
    t = rng.exponential(upper_limit/count, size=count).cumsum()
    return t, np.column_stack([t/(1.+t)**2, 1./(1.+t)**2])


def signal_b(count):
    t = rng.exponential(upper_limit/count, size=count).cumsum()
    return t, np.column_stack([np.cos(t), np.sin(t)])


params_a, true_signal_a = signal_a(100)
params_b, true_signal_b = signal_b(100)

fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots(2, 2, tight_layout=True)

ax11.plot(params_a, true_signal_a[:, 0], "k")
ax11.plot(params_a, true_signal_a[:, 1], "k--")
ax11.legend(["x", "y"])
ax12.plot(params_b, true_signal_b[:, 0], "k")
ax12.plot(params_b, true_signal_b[:, 1], "k--")
ax12.legend(["x", "y"])
ax21.plot(true_signal_a[:, 0], true_signal_a[:, 1], "k")
ax22.plot(true_signal_b[:, 0], true_signal_b[:, 1], "k")
ax11.set_title("Components of signal a")
ax11.set_xlabel("Parameter")
ax11.set_ylabel("Value")
ax12.set_title("Components of signal b")
ax12.set_xlabel("Parameter")
ax12.set_ylabel("Value")
ax21.set_title("Signal a")
ax21.set_xlabel("x")
ax21.set_ylabel("y")
ax22.set_title("Signal b")
ax22.set_xlabel("x")
ax22.set_ylabel("y")

plt.show()

signature_a = esig.stream2sig(true_signal_a, 2)
signature_b = esig.stream2sig(true_signal_b, 2)
print(signature_a, signature_b, sep="\n")


sigs_a = np.vstack([esig.stream2sig(make_noisy(signal_a(rng.integers(50, 100))[1]), depth) for _ in range(50)])
sigs_b = np.vstack([esig.stream2sig(make_noisy(signal_b(rng.integers(50, 100))[1]), depth) for _ in range(50)])

expected_sig_a = np.mean(sigs_a, axis=0)
expected_sig_b = np.mean(sigs_b, axis=0)
print(expected_sig_a, expected_sig_b, sep="\n")

diff = np.abs(expected_sig_a - expected_sig_b)

print("Signal a", np.max(np.abs(expected_sig_a - signature_a)))
print("Signal b", np.max(np.abs(expected_sig_b - signature_b)))
print("Signal a vs signal b", np.max(np.abs(expected_sig_a - expected_sig_b)))

