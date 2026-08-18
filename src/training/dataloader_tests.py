import torch
from torchvision.transforms import ToTensor
from torch import nn
from evaluation_metrics import roc_curve
from training import data_handling
import matplotlib.pyplot as plt

dataset = data_handling.DataStructure()
dataset.load_test_data(7, 32, "/home/sebastien/Project/TB_Prediction_Model/src/training/test_data/fashionM")



dataloader = dataset.dataloaders[0]

image_batch, label_batch = next(iter(dataloader))

for i in range(len(image_batch)):
    print(label_batch[i])
    plt.imshow(image_batch[i].squeeze(), cmap="gray")
    plt.show()


#
# class TestModel(nn.Module):
#     def __init__(self, input_shape, hidden_nodes, output_shape):
#         super().__init__()
#         self.layers = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(input_shape, hidden_nodes),
#             nn.Linear(hidden_nodes, hidden_nodes),
#             nn.Linear(hidden_nodes, output_shape)
#         )
#
#     def forward(self, x):
#         return self.layers(x)


#
# shape = next(iter(train_dataloader))[0][0].squeeze().shape
# input_vector_size = shape[0]*shape[1]
#
# test_model = TestModel(input_vector_size, 64, 1)
# test_model.to("cpu")
#
# optimizer = torch.optim.SGD(params=test_model.parameters())
# loss_function = nn.CrossEntropyLoss()
#
# epochs = 5
#
# test_model.train()
#
#
#
#
# with torch.inference_mode():
#     test_model.eval()
#     for bach, (x,y) in enumerate(test_dataloader):
#         test_pred = test_model(x).squeeze()
#         roc_curve.draw_roc_curve(list(torch.sigmoid(test_pred)), y)

