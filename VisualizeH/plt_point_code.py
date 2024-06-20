import torch
from VisualizeH.tsne import tsne
import numpy as np
from VisualizeH.distance_distribution.intra_inter_distance import displt
tensor1 = torch.load("/home/gml/HXC/images/point/T/BEST/tensor1.pt")
tensor2 = torch.load("/home/gml/HXC/images/point/T/BEST/tensor2.pt")
tensor3 = torch.load("/home/gml/HXC/images/point/T/BEST/label.pt")
a1=tensor1[0:1000]
a2=tensor2[0:1000]
a3=tensor3[0:1000]
# for i in range(60):
#     a1[4*i:4*i+4]=tensor1[i*360:i*360+4]
#     a2[4*i:4*i+4]=tensor2[i*360:i*360+4]
#     a3[4*i:4*i+4]=tensor3[i*360:i*360+4]

# tsne.plot_embedding_2d(a1.cpu().data.numpy(),a2.cpu().data.numpy(),
#                        a3.cpu().data.numpy())
displt(tensor1,tensor3,tensor2,tensor3)

print("")