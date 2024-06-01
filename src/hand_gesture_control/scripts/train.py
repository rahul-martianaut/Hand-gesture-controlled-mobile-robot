import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils.model_keypoint import KeyPointClassifier_model



# Define constants
RANDOM_SEED = 42
NUM_CLASSES = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = 'model/keypoint_classifier/keypoint.csv'
#model_save_path = 'model/keypoint_classifier/keypoint_classifier.hdf5'
#tflite_save_path = 'model/keypoint_classifier/keypoint_classifier.tflite'

# Load dataset
X_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))
y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))
X_train, X_temp, y_train, y_temp = train_test_split(X_dataset, y_dataset, train_size=0.75, random_state=RANDOM_SEED)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size=0.5, random_state=RANDOM_SEED)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader
train_data = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

val_data = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_data, batch_size=128, shuffle=True)

test_data = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_data, batch_size=128, shuffle=True)


# Training loop
def train(model, loss_fn, optimizer):
    num_epochs = 1000
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()            
            running_loss += loss.item()
    
        if (epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch+1, num_epochs, running_loss, (100 * correct / total)))

    print('Finished Training')

# Save the trained model
#torch.save(model.state_dict(), 'model/keypoint_classifier/pytorch_keypoint_classifier.pth')

def validation(model, loss_fn):  # VALIDATION
    model.eval()  # evaluation mode
    validation_loss = 0
    validation_acc = 0
    total_test = 0
    correct_test = 0
    val_loss = 0

    with torch.no_grad():
        #val_loop = tqdm(val_loader, leave=True)
        for val_input, val_label in val_loader:
            val_input, val_label = val_input.to(device), val_label.to(device)

            val_output = model(val_input)
            # print("out ",val_output)
            # print("val_label ", val_label)
            loss = loss_fn(val_output, val_label)
            val_loss += loss.item() * val_input.size(0)
            _, predicted = torch.max(val_output.data, 1)
            total_test += val_label.size(0)
            correct_test += (predicted == val_label).sum().item()

    validation_acc += (100 * correct_test / total_test)
    validation_loss += (val_loss / len(val_loader))

    return validation_loss, validation_acc

# Evaluate on test set
def test(model):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy on test set: {:.2f}%'.format(100 * correct / total))


def main():

    # Initialize model
    model = KeyPointClassifier_model(input_size=21*2, num_classes=NUM_CLASSES).to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    TRAIN = False
    TEST = True

    if TRAIN:
        train(model, loss_fn, optimizer)
        val_loss, val_acc = validation(model, loss_fn)
        best_acc=0
        if val_acc > best_acc:  # best accuracy out of all epochs
            best_acc = val_acc
            torch.save(model.state_dict(), 'model/keypoint_classifier/pytorch_keypoint_classifier.pth')
            # print("val_acc: ", val_acc)

    if TEST:
        model.load_state_dict(torch.load('model/keypoint_classifier/pytorch_keypoint_classifier.pth'))
        test(model)



if __name__ == "__main__":
    main()

