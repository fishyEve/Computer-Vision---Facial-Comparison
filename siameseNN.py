import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import face_recognition
from itertools import combinations
import random
random.seed(None)
np.random.seed(None)
# Eve Collier
# CPSC 372 - Independent Study
# Project 2: Facial Verification

# generate_pairs
# Helper function to generate image pairs for training our Siamese neural network.
def generate_pairs(imgData='ImageBasics'):
# Resource: https://pyimagesearch.com/2020/11/23/building-image-pairs-for-siamese-networks-with-python/
    images = []      # Tuples of (image, face encoding)
    imgPaths = []    # File paths for every image
    
    # Go thru all images in the folder
    # https://www.geeksforgeeks.org/python-os-listdir-method/ 
    for filename in os.listdir(imgData):
        # Only interested in pictures
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(imgData, filename)
            # https://pypi.org/project/face-recognition/
            image = face_recognition.load_image_file(path)     # Load image
            encodings = face_recognition.face_encodings(image) # Get the face encodings
            
            if encodings:  # If face_recognition detects a face...
                images.append((image, encodings[0]))  # Save it w/ it's encoding 
                imgPaths.append(path)                 # Annnd save the file path to access image again 

    random.shuffle(images) # Randomize grouping
    groups = []            # Groups of similar faces
    grouped = set()        # To keep track of which images fall in a group
    
    # Group together similar faces
    for i, (_, face1) in enumerate(images):
        if i in grouped: # If we've already grouped this one...
            continue  # Skip
            
        group = [i] # Start new group w/ current face
        for j, (_, face2) in enumerate(images[i+1:], i+1):
            if j in grouped: # If we've already grouped this one...
                continue  # Skip

            # If the Euclidean distance of these two is less then .6, the face belongs to the same person
            if np.linalg.norm(face1 - face2) < 0.6:
                group.append(j)  # Add to group
                grouped.add(j)   # Mark face as grouped
        
        groups.append(group) # Save the group
        grouped.add(i)       # Mark current face as grouped 

    random.shuffle(groups) # Shuffle order of all our groups 
    pairs = []   # Pairs of face encodings for face_recognition
    labels = []  # Labels: 0 = same person, 1 = different people
    
    # Generate pairs of face encodings from the same person (AKA our positive pairs)
    for group in groups:
        if len(group) >= 2: # Must have at least 2 faces per group
            for i, j in combinations(group, 2): # ALL possible pairs in group
                pairs.append((images[i][1], images[j][1])) # Add encodings
                labels.append(0) # Add our label, 0 = same
        
    # And now to generate pairs from different people (AKA our negative pairs)
    group_indices = list(range(len(groups)))
    for _ in range(len(pairs)):  # To fix err- need same num of positive and negative pairs
        if len(groups) < 2: 
            break # Must be at least two groups
            
        # Randomly pick from two different groups 
        g1, g2 = np.random.choice(group_indices, 2, replace=False)
        i1 = np.random.choice(groups[g1]) # Random face from group 1
        i2 = np.random.choice(groups[g2]) # Random face from group 2
        pairs.append((images[i1][1], images[i2][1])) # Add face encodings for face_recognition 
        labels.append(1) # Add our label, 1 = different
    
    return pairs, labels, imgPaths

# SiameseNetwork
# Class definition for our Siamese Neural Network design
class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.lay1 = nn.Linear(128, 256) # Input 128 (face encodings), 256 output for hidden layer
        self.bn1 = nn.BatchNorm1d(256)  # Batch normalization- idk exactly what this is but https://medium.com/hackernoon/facial-similarity-with-siamese-networks-in-pytorch-9642aa9db2f7 they did it
        #self.bn1 = nn.LayerNorm(256)   # Mentioned in class this is better then batch normalization, will test l8r
        self.dop = nn.Dropout(0.5)      # Dropout- same deal as the batch normalization, a good few tutorials used this as well
        self.out = nn.Linear(256, 64)   # Output layer, these values were the ones that worked
        #self.out = nn.Linear(128, 64)
        #self.relu = nn.ReLI(inplace=True)
        #self.lay1 = nn.Linear(128, 512)
        #self.bn1 = nn.BatchNorm1d(512)
        #self.drop1 = nn.Dropout(0.5)
        #self.lay2 = nn.Linear(512, 256)
        #self.bn2 = nn.BatchNorm1d(256)
        #self.drop2 = nn.Dropout(0.3)
        #self.out = nn.Linear(256, 128)
    
    def forward(self, x):
        #x = F.relu(self.bn1(self.lay1(x)))
        x = self.dop(F.relu(self.bn1(self.lay1(x))))
        #return torch.sigmoid(self.out(x))  # Constrain outputs to [0,1], made every face the same 
        return F.normalize(self.out(x), p=2, dim=1)  # Normalized embeddings
# ContrastiveLoss
# Custom loss function for our Siamese Neural Network. We want to minimize the distance for similar face pairs 
# while maximizing distance for non-similar face pairs
class ContrastiveLoss(nn.Module):
#https://jamesmccaffrey.wordpress.com/2022/03/04/contrastive-loss-function-in-pytorch/ must override loss function
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin  # Value for what makes pairs not similar
    
    def forward(self, output1, output2, label):
        #euclidean_distance = F.pairwise_distance(output1, output2)
        #loss = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                         #label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        #return loss
        #euclidean_distance = F.pairwise_distance(output1, output2)
        #pos_loss = label * torch.pow(euclidean_distance, 2)
        #neg_loss = (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        #return torch.mean(pos_loss + neg_loss)
        distance = F.pairwise_distance(output1, output2) # Euclidean distance
        loss = (1-label)*distance.pow(2) + label*F.relu(self.margin-distance).pow(2) # Active for positive and negative pairs
        return loss.mean() # Average loss

# FaceDataset
# Custom PyTorch Dataset class to load pairs and labels to train our network
class FaceDataset(Dataset):
    def __init__(self, pairs, labels):
        self.pairs = pairs   # List of embeddings
        self.labels = labels # List of labels
    
    def __len__(self):
        return len(self.pairs) # Total num of pairs
    
    # Get corresponding pair of face encodings + label
    # Convert to tensor for PyTorch to understand it. 
    def __getitem__(self, idx):
        return torch.FloatTensor(self.pairs[idx][0]), \
               torch.FloatTensor(self.pairs[idx][1]), \
               torch.FloatTensor([self.labels[idx]])

def train_model(model, train_loader, epochs=50, lr=0.0001):
    # https://pyimagesearch.com/2020/12/07/comparing-images-for-similarity-using-siamese-networks-keras-and-tensorflow/ 
    criterion = ContrastiveLoss(margin=1.0) # Call loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # Optimizer- used in previous MIT project
    #model.train()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for data1, data2, label in train_loader:
                optimizer.zero_grad() # Reset gradients 
                output1, output2 = model(data1), model(data2)  # Forward pass
                loss = criterion(output1, output2, label)      # Get the loss
                loss.backward()  # Backprop
                optimizer.step() # Update our weights
                running_loss += loss.item() # Update running loss
    
    #for epoch in range(epochs):
        #model.train()
        #running_loss = 0.0
        #for data1, data2, label in train_loader:
            #optimizer.zero_grad()
            #output1, output2 = model(data1), model(data2)
            #loss = criterion(output1, output2, label)
            #loss.backward()
            #optimizer.step()
            #running_loss += loss.item()
        #for batch_idx, (data1, data2, label) in enumerate(train_loader):
            #optimizer.zero_grad()
            #output1 = model(data1)
            #output2 = model(data2)
            #loss = criterion(output1, output2, label)
            #loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Clip gradients
            #optimizer.step()
            #running_loss += loss.item()
        
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}')
    
    return model

def compare_faces(model, embedding1, embedding2, threshold=0.5):
    model.eval() # Set the model to evaluation mode
    with torch.no_grad(): # Disable gradient computation
        emb1 = model(torch.FloatTensor(embedding1).unsqueeze(0)) # Get embedding 1
        emb2 = model(torch.FloatTensor(embedding2).unsqueeze(0)) # Get embedding 2
        distance = F.pairwise_distance(emb1, emb2).item() # Compute distance between them
        #threshold = 0.65 if distance < 1.0 else 1.25
        return distance < threshold, distance # True if its the same person, false if otherwise, 
                                              # we also return the distance val.
    
