import numpy as np

def escale(data, interval=(0., 1.)):
  a, b = interval
  max, min = np.max(data), np.min(data)
  if max == min:
    return data * 0
  escaled_data = (b - a) * ((data - min) / (max - min)) + a
  return escaled_data

def escale_all(data, interval=(0., 1.)):
  escaled_data = np.apply_along_axis(escale, 0, data)
  return escaled_data

def z_score(data):
  standarize = (data - np.mean(data)) / np.std(data)
  return standarize

def z_score_all(data):
  standarize = np.apply_along_axis(z_score, 0, data)
  return standarize