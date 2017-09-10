import numpy as np


class CharRNN:
  
  # Input:  x:  255(asci) x 1
  # Output  o_t: 255(asci) x 1
  # State:  s_t: T x 1
  # Weights: T x 255(asci)
  # U and V: 255(asci) x T

  def __init__(self, char_dim, hidden_dim, bptt_truncate):
    self.char_dim = char_dim 
    self.hidden_dim = hidden_dim
    self.bptt_truncate = bptt_truncate

    #random initialize parameters
    self.U = np.random.uniform(-np.sqrt(1./char_dim), np.sqrt(1./char_dim), (hidden_dim, char_dim))
    self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (char_dim, hidden_dim))
    self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))

  def forward_prop(self, x):
    T = len(x)
    s = np.zeros((T+1, self.hidden_dim))
    #s[-1] = np.zeros(self.hidden_dim) 

    for t in np.arange(T):
      s[t] = np.tanh(self.U[:, x[t]] + self.W.dot(s[t-1])
      o[t] = softmax(self.V.dot(s[t])
    rxeturn [o, s]
