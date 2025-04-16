# FACE COMPARISON USING COMPUTER VISION
Eve Collier

CPSC 372 - Spring 2025

Independent Study Project 2 - Computer Vision

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




## DEVELOPMENT PROCESS
This project was hard, as I've never started with an empty file and went 'okay, time to make a neural network'. Luckily,
there were a lot of resources and examples online of neural networks (even siamese ones) implemented with PyTorch. At first,
I was going to do a MIT project for my second project for my independent study, but certain URLs weren't working and I decided
to just make my own project. The MIT project was working toward using computer vision for facial recognition, so I started looking 
online and found there were already a lot of ways, and even a Python library, for facial recognition. I decided to use the library 
and take things a step forward- train a network to compare two different recognized faced and determine whether or not the faces 
come from the same person.

I started this off by researching neural networks that were good with comparisons. Naturally, I discovered Siamese Networks and 
decided to implement since I could use it to learn similarities. Examples of Siamese Networks use overridden Contrastive Loss methods,
so I decided to do that as well. I wrote the code for my neural network first, then I wrote the implementation of it (and discovered
everything that was wrong with my neural network).

One of the first problems I ran into was not having balanced positive and negative pairs during training- sometimes I'd wind up with 
all of one and none of the other. This was when I decided to implement the generate_pairs() helper function to group similar faces together 
using the Euclidean distance- before I was generating the pairs randomly (and somehow would end with all of one and none of the other) and 
not by similarity. When I grouped by similarity (which makes sense since we are trying to compare images, I don't know why I didn't do this
in the first place) the number of positive vs negative pairs stopped being concerningly different from one another.

I had a lot of problems with the images at first- I had to do a lot of research to figure out how to use OpenCV. There was a really 
annoying problem I had with the faces were misaligned which I felt was throwing off my results. I managed to fix this with:
aligned_face = dlib.get_face_chip(img, face_landmarks)
which is a single line of code that I originally was writing lines upon lines for.

My training method for the network (obviously) didn't work at first. I got to play the fun game of setting random learning rates,
trying different PyTorch optimizers (but looping back to the one I started with, Adam, in the end), trying different epochs (50 
worked), and all of that fun stuff. I can't accurately described what I changed because I was changing things over the course of a 
week and loosing my mind in the process.

I had to figure out how to interface with the face_recognition Python library. A lot of things confused me at first, with the main 
thing being the embeddings. One thing that was super fun was the fact that face_recognition gives it's embeddings in RGB format, 
later on when I wanted to use OpenCV I tried using the data in it's original, RGB format and things were going way wrong. The conversion 
from RGB to BGR wasn't hard to write, but it did take me a second to realize that this was an issue and a reason why my program wasn't
running at first. I also had to figure out the translation of the data between face_recognition to a format for PyTorch 
to understand (PyTorch tensors). 

There were additional issues I ran into other than what's listed above. I'd add them to the list, but I've managed to block out of my consciousness.

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


