import numpy as np


class CharRNN:
  
  # Input:  x:  255(asci) x 1
  # Output  o_t: 255(asci) x 1
  # State:  s_t: T x 1
  # Weights: T x 255(asci)
  # U and V: 255(asci) x T

  def __init__(self, char_dim, hidden_dim):
    self.char_dim = char_dim 
    self.hidden_dim = hidden_dim
    
    #random initialize parameters with the 1/nsqrt init
    self.U = np.random.uniform(-np.sqrt(1./char_dim), np.sqrt(1./char_dim), (hidden_dim, char_dim))
    self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (char_dim, hidden_dim))
    self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))

  def forward_prop(self, x):
    T = len(x)
    print T
    s = np.zeros((T+1, self.hidden_dim))
    #s[-1] = np.zeros(self.hidden_dim) 
    o = np.zeros((T, self.char_dim))
    for t in np.arange(T):
      s[t] = np.tanh(self.U[:, x[t]] + self.W.dot(s[t-1]))
      print s[t]
      out = self.V.dot(s[t])
      print out.shape
      o[t] = self.softmax(out)
    return [o, s]

  def softmax(self, x):
    x_exp = np.exp(x - np.max(x)) # numerical stability to avoid inf
    return x_exp/np.sum(x_exp) 

  def predict(self, x):
    o, s = self.forward_prop(x)
    return np.argmax(o, axis=1)
  
if __name__ == "__main__":
  arr = np.array('hello', 'c')
  arr_int = arr.view(np.uint8)
  print arr_int
  rnn_model = CharRNN(255, 10)
  prediction =  rnn_model.predict(arr_int)
  print prediction
  print [chr(ch) for ch in prediction.tolist()]
