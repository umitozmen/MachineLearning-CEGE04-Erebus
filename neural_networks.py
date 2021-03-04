import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Load_Ceserian_Dataset import readCeserianFile
from keras.utils import to_categorical
import torch.nn.functional as F


data = readCeserianFile("caesarian.csv.arff")


class Model(nn.Module):
    def __init__(self, input_dim):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_dim, 50)
        self.layer2 = nn.Linear(50, 20)
        self.layer3 = nn.Linear(20, 2)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x))  # To check with the loss function
        return x


features =np.array(data[['Age','Delivery number', 'Delivery time','Blood of Pressure', 'Heart Problem']]).astype(np.float)
labels =np.array(data['Caesarian']).astype(np.float)
ass = features[1][1]
print (features, labels, features[1][1])


features_train,features_test, labels_train, labels_test = train_test_split(features, labels, random_state=42, shuffle=True)


# Training
model = Model(features_train.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()
epochs = 100

def print_(loss):
    print ("The loss calculated: ", loss)


# Not using dataloader
x_train, y_train = Variable(torch.from_numpy(features_train)).float(), Variable(torch.from_numpy(labels_train)).long()
for epoch in range(1, epochs + 1):
    print("Epoch #", epoch)
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)
    print_(loss.item())

    # Zero gradients
    optimizer.zero_grad()
    loss.backward()  # Gradients
    optimizer.step()  # Update


# Prediction
x_test = Variable(torch.from_numpy(features_test)).float()
pred = model(x_test)

pred = pred.detach().numpy()
pred


print ("The accuracy is", accuracy_score(labels_test, np.argmax(pred, axis=1)))

# Checking for first value
np.argmax(model(x_test[0]).detach().numpy(), axis=0)