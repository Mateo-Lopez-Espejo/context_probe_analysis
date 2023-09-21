import numpy as np

from scipy.io import wavfile
from nems.analysis.gammatone.gtgram import gtgram


def calculate_sound_metrics(soundfile):
    """
    Calculates a set of spectro-temporal metrics out of the first second of a
    .wav file.
    The metrics calculated are:
    1. Bandwidth, containing the 70% of the spectral power of the sound.
    2. Spectral Correlations, averaged across all spectral bin correlations
    3. Temporal Stationarity, the average standard deviation over time.
    Args:
        soundfile: Path to a .wav file

    Returns:
        bandwidth: Float
        average_spectral_correlation: Float
        temporal_stationarity: Float
        spectrogram: 2d numpy array

    """
    # filtering parameters
    lfreq, hfreq, bins = 100, 24000, 48

    # Loads wave from .wav file and transfrorms into its spectrogram
    # representation with dimension spectrogram of
    # Frequency (axis 0) x Time (axis 1)
    sfs, W = wavfile.read(soundfile)
    W = W[:sfs]  # cut first second of the sound as it was the only part used
    spectrogram = gtgram(
        W, sfs, 0.02, 0.01, bins, lfreq, hfreq
    )

    #### Bandwidth ####
    # Number of octaves that contain between 15% and 85% of the spectral power.
    # This interval was selected empirically.
    lower, upper = [0.15, 0.85]

    # Get the cumulative sum of spectral power to then get the quantile powers
    power_spectrum = np.nanmean(spectrogram, axis=1)
    cumulative_power = np.cumsum(power_spectrum)
    total_power = np.max(cumulative_power)

    # Find the bins corresponding to the 15 and 85 percentiles\
    bin_high = np.abs(cumulative_power - (total_power * upper)).argmin()
    bin_low = np.abs(cumulative_power - (total_power * lower)).argmin()

    # converts quantile bins into frequencies and their difference into octaves
    spectral_bins = np.logspace(np.log2(lfreq), np.log2(hfreq), num=bins,
                                base=2)
    bandwidth = np.log2(spectral_bins[bin_high] / spectral_bins[bin_low])

    # We can use the bandwidth to filter the spectrogram and calculate other
    # metrics over frequency bins with some power in them.
    bw_spectrogram = spectrogram[bin_low:bin_high, :]

    ##### Spectral correlations #####
    # How much are the frequency channels correlated between them
    # and what is the average of these correlations.
    spectral_correlations = np.corrcoef(bw_spectrogram)
    average_spectral_correlation = spectral_correlations[
        np.triu_indices(bw_spectrogram.shape[0], k=1)].mean()

    ##### Temporal Stationarity #####
    # How much frequency variation changes over time.
    # This is done by taking the standard deviation (over time) of each
    # frequency bin, and then calculating the average (over frequency) of these
    # deviations. Done over the filtered spectrogram to avoid spectral channels
    # with no power and therefore no variation.
    temporal_stationarity = np.nanmean(np.std(bw_spectrogram, axis=1))

    return (
        bandwidth,
        average_spectral_correlation,
        temporal_stationarity,
        spectrogram
    )
