import keyboard
# Eve Collier
# CPSC 372 - Spring 2025
# Independent Study Project 2 - Computer Vision
# I gave up on MIT. I'll do my own facial recognition. 

### SNIPPET FROM https://neptune.ai/blog/15-computer-visions-projects ###
import cv2
import face_recognition
import numpy as np

imgmain = face_recognition.load_image_file('ImageBasics/dude.jpeg')
imgmain = cv2.cvtColor(imgmain, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('ImageBasics/lady.jpeg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgmain)[0]
encodeElon = face_recognition.face_encodings(imgmain)[0]
cv2.rectangle(imgmain, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodeElon], encodeTest)
faceDis = face_recognition.face_distance([encodeElon], encodeTest)

# EVE ADDED Get the max height between the two images
max_height = max(imgmain.shape[0], imgTest.shape[0])

# EVE ADDED Resize both images to the same height (keeping aspect ratio)
imgmain_resized = cv2.resize(imgmain, (int(imgmain.shape[1] * max_height / imgmain.shape[0]), max_height))
imgTest_resized = cv2.resize(imgTest, (int(imgTest.shape[1] * max_height / imgTest.shape[0]), max_height))

### GEEKSFORGEEKS to get images side-by-side:
# concatenate image Horizontally 
Hori = np.concatenate((imgmain_resized, imgTest_resized), axis=1) 
  
print(results, faceDis)
cv2.putText(Hori, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

#cv2.imshow('Main Image', imgmain)
#cv2.imshow('Compared Image', imgTest)
cv2.imshow('Side-By-Side Comparison', Hori) 

cv2.waitKey(0)

### END OF CODE SNIPPET ##
if keyboard.is_pressed('n'):
    cv2.destroyAllWindows()
    #sys.exit()

# The snippet of code above compares two faces to one another and returns true or false based on whether or not
# those faces are identical.
# I will extend this a step forward and train a NN to learn face similarity. BOOM.

# https://www.geeksforgeeks.org/siamese-neural-network-in-deep-learning/
# https://en.wikipedia.org/wiki/Siamese_neural_network 
# Lets use a Siamese network! That sounds cool!

# Import PyTorch and other relevant libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np




