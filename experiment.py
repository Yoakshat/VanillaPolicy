from torch.distributions import Categorical
import torch

# assume ten actions
# 2x2x2

probLst = [[[0.4, 0.6], [0.7, 0.3]], [[0.1, 0.9], [0.5, 0.5]]]

pickActions = torch.tensor([[1, 0], [1, 0]], dtype=torch.long)
actProbs = torch.tensor(probLst)

m = Categorical(actProbs)

print(m.log_prob(pickActions))

