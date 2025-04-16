import cv2
import face_recognition
import numpy as np
import os
from siameseNN import SiameseNetwork, FaceDataset, train_model, compare_faces, generate_pairs
import torch
from torch.utils.data import DataLoader
from itertools import combinations

# Eve Collier
# CPSC 372 - Independent Study
# Project 2: Facial Verification



def visualize_comparison(img1, img2, is_match, distance):
    # Convert from RGB (from face_recognition) to BGR (for OpenCV) because my life wasn't hard enough before
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
    
    # Resize both images to have the same height - it looked silly before this 
    max_height = max(img1.shape[0], img2.shape[0])
    img1 = cv2.resize(img1, (int(img1.shape[1]*max_height/img1.shape[0]), max_height))
    img2 = cv2.resize(img2, (int(img2.shape[1]*max_height/img2.shape[0]), max_height))
    
    # Add result text 
    result_text = f"{'SAME' if is_match else 'DIFF'}: {distance:.2f}"
    font_scale = img2.shape[1] / 500  # Adjust divisor to get desired size
    thickness = max(2, int(font_scale * 2)) # Couldn't see text before this
    text_color = (0, 255, 0) if is_match else (0, 0, 255)  # Green for match, red for different
    
    # Calc text size and its position
    (text_width, text_height), _ = cv2.getTextSize(result_text, 
                                                  cv2.FONT_HERSHEY_SIMPLEX, 
                                                  font_scale, 
                                                  thickness)
    
    # Add padding to stay within image bounds
    text_x = 30
    text_y = text_height + 30  # Position text below top edge with padding
    
    # Add background rectangle - hard to see text otherwise
    cv2.rectangle(img2, 
                 (text_x - 10, text_y - text_height - 10), 
                 (text_x + text_width + 10, text_y + 10),  
                 (0, 0, 0), -1)  # Black background
    
    # Put text on image
    cv2.putText(img2, result_text, (text_x, text_y), 
               cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
    
    # Combine the two pictures
    comparison = np.concatenate((img1, img2), axis=1)
    cv2.imshow('Face Comparison', comparison)
    cv2.waitKey(3000)       # Show for 3 seconds
    cv2.destroyAllWindows() # Get rid of it and onto the next

# main()
# Runner code for entire program- loads face images, trains our Siamese NN, and runs the comparisons between faces
def main():
    # Prepare data from ImageBasics folder
    pairs, pair_labels, image_paths = generate_pairs()
    
    # Enforce at least 10 pairs- we need data to train the network
    if len(pairs) < 10:
        print(f"Need more images (found {len(pairs)} pairs). Add at least 4 images of 2+ people.")
        return
    
    # Initialize and train model
    model = SiameseNetwork()  # Our model instance
    dataset = FaceDataset(pairs, pair_labels) # Dataset
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True) # Data loader
    model = train_model(model, train_loader, epochs=20) # Train for 20 iterations (epochs)

    # DEBUG PRINT STATEMENTS
    print(f"Total pairs: {len(pairs)} (Pos: {pair_labels.count(0)}, Neg: {pair_labels.count(1)})")
    #sample_distances = [np.linalg.norm(p[0]-p[1]) for p in pairs[:5]]
    #print("Sample distances:", sample_distances)
    
    # Now we can compare the face images 
    print("\nTesting with images...")
    
    compareNum = min(25, len(image_paths)) # Go thru 25, that way we get at least one different pair and one same (hopefully)
    #compareNum = np.arange(len(image_paths))
    alreadyCompared = set()  # Keep track of faces we've compared so we don't get duplicate results
    # Keep trying until we get compareNum unique pairs
    while len(alreadyCompared) < compareNum:
        idx1, idx2 = np.random.choice(len(image_paths), 2, replace=False)
        sortedPair = tuple(sorted((idx1, idx2)))
        # Only proceed IF we haven't already compared this pair
        if sortedPair not in alreadyCompared:
            alreadyCompared.add(sortedPair) # Add this pair
            sortedPair = tuple(sorted((idx2, idx1))) # Invert it for other way around (faces being on other sides of screen)
            alreadyCompared.add(sortedPair) # Add the inverse of the pair- we don't wanna compare these two faces again period
        
            img1 = face_recognition.load_image_file(image_paths[idx1]) # Load first image w/ face_recognition
            img2 = face_recognition.load_image_file(image_paths[idx2]) # Load second image w/ face_recognition
        
            enc1 = face_recognition.face_encodings(img1)[0] # face encodings from first face 
            enc2 = face_recognition.face_encodings(img2)[0] # face encodings from second face 
        
            # Compare the two face encodings (AKA vector of face feats) 
            isMatch, distance = compare_faces(model, enc1, enc2)
            # Display the result of the comparison 
            visualize_comparison(img1, img2, isMatch, distance)



            
    #for _ in range(compareNum):
    #for i in range(0, min(compareNum*2, len(image_paths)), 2):
        #idx1, idx2 = np.random.choice(len(image_paths), 2, replace=False)
        #idx1, idx2 = compareNum[i], compareNum[i+1]
        #sortedPair = tuple(sorted((idx1, idx2)))
        #if sortedPair not in alreadyCompared:
            #alreadyCompared.add(sortedPair)
            #break
        #img1 = face_recognition.load_image_file(image_paths[idx1])
        #img2 = face_recognition.load_image_file(image_paths[idx2])
        
        #enc1 = face_recognition.face_encodings(img1)[0]
        #enc2 = face_recognition.face_encodings(img2)[0]
        
        #is_match, distance = compare_faces(model, enc1, enc2)
        #visualize_comparison(img1, img2, is_match, distance)


    #for i, path in enumerate(image_paths):
        #img = face_recognition.load_image_file(path)
        #enc = face_recognition.face_encodings(img)[0]
        
        # Compare with next image
        #if i < len(image_paths)-1:
            #next_img = face_recognition.load_image_file(image_paths[i+1])
            #next_enc = face_recognition.face_encodings(next_img)[0]
            
            #is_match, distance = compare_faces(model, enc, next_enc)
            #visualize_comparison(img, next_img, is_match, distance)

if __name__ == "__main__":
    main()