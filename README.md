# FACE COMPARISON USING COMPUTER VISION

## FILE MANIFEST
**siameseNN.py - This file is the implementation of a Siamese Neural Network for face verification/comparison. Here is a summary of the key components:**

1. generate_pairs() helper function
   - Loads images from a folder containing our face data
   - Based on the Euclidean distance, we group together similar faces (with a threshold of 0.6)
   - Generate pairs of positive (same faces) pairs and negative (different faces) pairs
   - Returns...
     - pairs: Tuples containing our face encodings (vector containing facial features)
     - labels: 0/1 for positive/negative pairs 
     - imgPaths: File paths for our images of faces
2. SiameseNetwork class
   - Defines the architecture of our siamese neural network:
     - Input layer: 128 dimensional face encodings (numbers that represent face feats- generated from face_recognition)
     - Input Layer(128D)->256D->BatchNorm->Dropout(0.5)->Output Layer(64D)
     - Output layer: the normalized version of our 64-dimensional vectors generated in the network. It contains the network's learned representations of
       facial features. The closer the embeddings are in this vector space, the more similar the learned facial features are to one another. The farther
       they are- the more different.
3. ContrastiveLoss class
   - Overridden custom loss function for our Siamese neural network:
     - We need to minimize the distance for positive pairs to teach the network similarity
     - We need to maximize the distance for negative pairs to teach the network dissimilarity
     - Returns...
       - loss.mean() - the average loss
4. FaceDataset class
   - PyTorch Dataset wrapper for face pairs w/ their labels
   - Returns three PyTorch tensors, one for each training iteration:
     - embedding1 - first face's 128D encoding (torch.FloatTensor)
     - embedding2 - second face's 128D encoding (also torch.FloatTensor)
     - label - 0 indicating we have two faces from the same person or 1 indicating we have
       faces from two seperate people (another torch.FloatTensor)
     - We use tensors for PyTorch (it's PyTorch's required data format for training)
5. train_model() helper function
   - Trains the network using Adam optimizer and calling our overriden ContrastiveLoss method
   - The network's goal of training is to learn differentiating face embeddings
   - Returns...
     - (The now trained) model: the siamese neural network with weights that map faces to differentiating
       embeddings
6. compare_faces() helper function
   - Compares two facial embeddings with one another using our trained model to decide if they match
   - Returns...
     - Match status (0/1 AKA true/false)
     - (Image's) distance: the computed distance score of the two faces in our output layer's vector 





**faceComparison.py - Main script for running face verification using a Siamese Neural Network. Calls the 
appropriate functions to handle training the network, having the network compare the images, and then 
visualization using OpenCV for image processing/display. Here is a summary of the key components:**

1. visualize_comparison helper function
   - displays two images, side-by-side, with annotations related to the network's findings.
   - Inputs:
     - img1 - First compared face image in RGB format
     - img2 - Second compared face image in RGB format
     - is_match - a boolean set to True if the two images are found to be from the same person, false otherwise
     - distance - the calculated similarity score
   - Each comparison is displayed on the screen for three seconds before transitioning to the next
2. main()
   - Backbone of the program- calls all the appropriate functions and ultimately runs the entire program



**ImageBasics**

A folder containing jpg, jpeg, and png images of faces. The data we will ultimately be training our network with.


## PROJECT DEPENDENCIES 
* face_recognition - face detection
* opencv-python    - image processing and visualization
* numpy            - math
* torch            - PyTorch 



## PROTOCOL
1. Data Prep
   - Call generate_pairs() to create labeled face pairs to train our network.
2. Training
   - Initalization
     - Create FaceDataset from our face pairs and their labels
     - Set up DataLoader with a batch size of 64 and shuffle set to 'True' so the images are in a different order
   - Actual training
     - 20 epochs
     - Compute embeddings for both of the images in the pair
     - Penalty for pairs that are similar and far apart
     - Penalty for pairs that are not similar and close together
     - Update weights with the Adam optimizer
3. Face Comparison
   - For each pair:
     - Load both images and extract their 128D encodings
     - Pass the encodings through the trained model to get the 64D embeddings (for PyTorch to understand)
     - Compute Euclidean distance between embeddings
     - Faces are classified as the same if distance < 0.5 (the threshold)
4. Visualization
   - Display the images on the screen, side-by-side with text indicating "SAME" in green (if match) or "DIFF" in red (non-match)
     with the Euclidean distance between the two face embeddings


## BUILD INSTRUCTION
Ensure you have siameseNN.py, faceComparison.py, and the ImageBasics folder in a repository. On the command line, simply run:

**python3 faceComparison.py**

From there all you have to do is watch the program go through and compare some of the images from the file, displaying on the screen 
whether or not they're the same or different with the computed Euclidean distance. You can add your own faces to ImageBasics as well 
to test it out- just be sure to include faces from at least two seperate people and at least four pictures per face.

## KNOWN ISSUES
- Sometimes false negatives occur (two faces being marked as 'DIFF' when they are, in fact, from the same person). It doesn't happen
  very often, and it's interesting because sometimes the faces from the same person that were marked as 'DIFF' will sometimes be
  correctly marked 'SAME' on other runs of the program.
- My face_comparison method seems to only work for very, very good headshots/pictures. I tried supplying my own photos of faces (of my
siblings) and it ran nowehere near as smoothly as it did when I used perfectly still photos of celebrities 

## WORKS CITED/UTILIZED RESOURCES
https://pypi.org/project/face-recognition/ 

https://builtin.com/machine-learning/siamese-network

https://pyimagesearch.com/2021/01/18/contrastive-loss-for-siamese-networks-with-keras-and-tensorflow/ 

https://towardsdatascience.com/how-to-train-your-siamese-neural-network-4c6da3259463/

https://pyimagesearch.com/2020/12/07/comparing-images-for-similarity-using-siamese-networks-keras-and-tensorflow/ 

https://pyimagesearch.com/2020/11/23/building-image-pairs-for-siamese-networks-with-python/

https://datahacker.rs/019-siamese-network-in-pytorch-with-application-to-face-similarity/

https://becominghuman.ai/siamese-networks-algorithm-applications-and-pytorch-implementation-4ffa3304c18

https://medium.com/hackernoon/facial-similarity-with-siamese-networks-in-pytorch-9642aa9db2f7

https://jamesmccaffrey.wordpress.com/2022/03/04/contrastive-loss-function-in-pytorch/


