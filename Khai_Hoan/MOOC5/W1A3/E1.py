from utils import *

### YOU CANNOT EDIT THIS CELL

model = djmodel(Tx=30, LSTM_cell=LSTM_cell, densor=densor, reshaper=reshaper)

output = summary(model) 