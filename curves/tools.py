import numpy as np

from curves.t_rate_list_format import TimeSeriesMerged


def high_pass(mergedlogged: TimeSeriesMerged):
    """applies a filter that is linear with abs(frequency),
    the units of the filtered data is 1/d where d is the unit of the original data"""
    geometrical_mean_logged = 0.
    for curve in mergedlogged.as_np:
        geometrical_mean_logged += curve
    geometrical_mean_logged /= len(mergedlogged.currencies)

    for currency in mergedlogged.currencies:
        mergedlogged[currency] = mergedlogged[currency] - geometrical_mean_logged
    high_passed: TimeSeriesMerged = mergedlogged.copy()
    for currency, curve in zip(high_passed.currencies, high_passed.as_np):
        comeback = np.concatenate((curve, np.flip(curve, axis=0)))
        # to remove discontinuity of the periodic boundary condition
        freqs = np.fft.fftfreq(len(comeback), d=1)
        filter = np.abs(freqs)
        comeback = np.fft.ifft(np.fft.fft(comeback) * filter)
        curve = comeback[:len(high_passed.merged_times)]
        high_passed[currency] = curve.real
    return high_passed


def high_pass_cutoff(mergedlogged: TimeSeriesMerged, cutoff: float):
    """applies a filter that is linear with abs(frequency),
    the units of the filtered data is 1/d where d is the unit of the original data"""
    geometrical_mean_logged = 0.
    for curve in mergedlogged.as_np:
        geometrical_mean_logged += curve
    geometrical_mean_logged /= len(mergedlogged.currencies)

    for currency in mergedlogged.currencies:
        mergedlogged[currency] = mergedlogged[currency] - geometrical_mean_logged
    high_passed1: TimeSeriesMerged = mergedlogged.copy()
    low_passed2: TimeSeriesMerged = mergedlogged.copy()
    for currency, curve in zip(high_passed1.currencies, high_passed1.as_np):
        comeback = np.concatenate((curve, np.flip(curve, axis=0)))
        # to remove discontinuity of the periodic boundary condition
        freqs = np.fft.fftfreq(len(comeback), d=1)
        filter1 = np.abs(freqs) > 1 / cutoff
        filter2 = np.abs(freqs) <= 1 / cutoff
        comeback1 = np.fft.ifft(np.fft.fft(comeback) * filter1)
        comeback2 = np.fft.ifft(np.fft.fft(comeback) * filter2)
        curve1 = comeback1[:len(high_passed1.merged_times)]
        curve2 = comeback2[:len(high_passed1.merged_times)]
        high_passed1[currency] = curve1.real
        low_passed2[currency] = curve2.real
    return high_passed1, low_passed2


def pca(high_passed, normalize=False):
    """principal component analysis
    parameter normalize True to account for required deposit
    returns ev(eigen-vals), P(basis), Pinv(inverted), covariance
    """
    covariance = np.cov(high_passed.as_np)
    _, P = np.linalg.eigh(covariance)
    order = np.argsort(_)
    order = np.flip(order)
    P = P[:, order]

    costeffectiveness = 2. / np.sum((np.abs(P)), axis=0, keepdims=True) if normalize else 1.
    P = P * costeffectiveness

    Pinv = np.linalg.inv(P)
    ev = np.diag(Pinv @ covariance @ P)

    # test = P @ np.diag(ev) @ Pinv
    return ev, P, Pinv, covariance


def gridify(priceLogged_percent: np.ndarray, profitRange_percent=0.5, accumulated_mvt=False, initIsMean=False):
    """

    :param priceLogged_percent:
    :param profitRange_percent:
    :param accumulated_mvt: should return accumulated_movement?
    :return: definition mask, defined values[, accumulated_movement]
    """
    """The hold quantity in presence of fluctuations in the price"""
    tmask = np.zeros(priceLogged_percent.size, dtype=bool)
    last = gridmid = np.mean(priceLogged_percent) if initIsMean else priceLogged_percent[0]
    gridified = []
    ok = True if initIsMean else False
    accu = 0.
    accuAbs = 0.
    for i, x in enumerate(priceLogged_percent):
        if gridmid - profitRange_percent / 2 > x:
            ok, gridmid = True, x + profitRange_percent / 2
            accuAbs += np.abs(priceLogged_percent[i] - last)
            accu += priceLogged_percent[i] - last
        elif gridmid + profitRange_percent / 2 < x:
            ok, gridmid = True, x - profitRange_percent / 2
            accuAbs += np.abs(priceLogged_percent[i] - last)
            accu += priceLogged_percent[i] - last
        last = x
        if ok:
            gridified.append(gridmid)
        tmask[i] = ok
    return (tmask, np.array(gridified)) + (accu, accuAbs) * accumulated_mvt


def derivate(plogged: np.ndarray, crossed_only=False, symmetrical_derivation=False):
    """

    :param plogged: logged curve in percentage
    :return: definition mask, defined values
    """
    center = np.mean(plogged)
    cross = [i for i, _ in enumerate(plogged[:-1]) if (plogged[i] - center) * (plogged[i + 1] - center) < 0]
    B = np.roll(plogged, -1) - np.roll(plogged, symmetrical_derivation * 1)
    tmask = np.zeros(plogged.size, dtype=bool)
    if crossed_only:
        tmask[cross[0]:cross[-1]] = True
    else:
        tmask[:] = True
    if symmetrical_derivation: tmask[0] = False
    tmask[-1] = False
    return tmask, B[tmask]


def profitability(msk1: np.ndarray, goal, msk2: np.ndarray, benchmarked):
    """

    :param msk1: np.ndarray[bool]
    :param goal: np.ndarray[float], typically the derivative
    :param msk2: np.ndarray[bool]
    :param benchmarked: np.ndarray[float], the amount of hold for the strategy
    :return:
    """
    if msk1.size != msk2.size:
        raise ValueError
    times_length = msk1.size
    msk = np.bitwise_and(msk1, msk2)
    test1 = np.zeros(times_length)
    test1[msk1] = goal
    test1 = test1[msk]
    test1 /= np.std(test1)
    test2 = np.zeros(times_length)
    test2[msk2] = benchmarked
    test2 = test2[msk]
    test2 /= np.std(test2)
    result = np.cov(test1, test2)
    return result[1, 0]
