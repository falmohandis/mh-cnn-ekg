import numpy as np
import pandas as pd



def extract_features_from_segment(segment, fs):
    features = {}

    # ---------------------------------------------
    # Locate peaks relying on NeuroKit labels
    # ---------------------------------------------
    peaks = {}
    for label in ['P', 'Q', 'R', 'S', 'T']:
        idxs = segment.index[segment['Label'] == label].tolist()
        peaks[label] = idxs[0] if idxs else np.nan

    # ---------------------------------------------------------------------
    # T-wave fallback logic: estimate T if missing
    # ---------------------------------------------------------------------
    t_is_estimated = False

    if np.isnan(peaks['T']) and not np.isnan(peaks['S']):
        start_idx = int(peaks['S'])
        end_idx = segment.index[-1]

        window = segment.loc[start_idx:end_idx, 'ECG_Clean']

        if len(window) > 0:
            # Estimate T as max absolute deflection after S
            t_relative_idx = np.argmax(np.abs(window.values))
            estimated_T_idx = window.index[t_relative_idx]

            # ---- CRITICAL FIX ----
            peaks['T'] = estimated_T_idx
            segment.loc[estimated_T_idx, 'Label'] = 'T'
            t_is_estimated = True

    features["T_estimated"] = int(t_is_estimated)
    
    # Try
    # 


    # -----------------------------
    # Interval length features
    # -----------------------------
    def time_diff(a, b):
        if np.isnan(a) or np.isnan(b):
            return np.nan
        return abs(b - a) / fs

    features["P_Q"] = time_diff(peaks['P'], peaks['Q'])
    features["Q_R"] = time_diff(peaks['Q'], peaks['R'])
    features["R_S"] = time_diff(peaks['R'], peaks['S'])
    features["S_T"] = time_diff(peaks['S'], peaks['T'])
    features["P_R"] = time_diff(peaks['P'], peaks['R'])

    # -----------------------------
    # Slope features
    # -----------------------------
    def slope(idx1, idx2):
        if np.isnan(idx1) or np.isnan(idx2):
            return np.nan
        y1 = segment.loc[idx1, 'ECG_Clean']
        y2 = segment.loc[idx2, 'ECG_Clean']
        return (y2 - y1) / ((idx2 - idx1) / fs)

    features["QR_slope"] = slope(peaks['Q'], peaks['R'])
    features["RS_slope"] = slope(peaks['R'], peaks['S'])

    # -----------------------------
    # Area under curve features
    # -----------------------------
    signal = segment['ECG_Clean'].values
    features["AUC_full"] = np.trapz(np.abs(signal), dx=1/fs)

    if not np.isnan(peaks['Q']) and not np.isnan(peaks['S']):
        qrs = segment.loc[peaks['Q']:peaks['S'], 'ECG_Clean']
        features["AUC_QRS"] = np.trapz(np.abs(qrs), dx=1/fs)
    else:
        features["AUC_QRS"] = np.nan

    return features
