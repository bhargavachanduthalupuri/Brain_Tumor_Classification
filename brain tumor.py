#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Function to load images from a folder
def load_images(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)ss
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (64, 64))
            images.append(img)
            labels.append(1 if 'yes' in filename else 0)  # Set label 1 for tumor images, 0 for normal images
    return images, labels

# Function to save images with predicted labels
def save_predicted_images(images, labels, predictions, output_dir):
    for i in range(len(images)):
        img = images[i]
        label = labels[i]
        prediction = predictions[i]

        # Add label and prediction text to the image
        text = f"Label: {'Tumor' if label == 1 else 'Normal'}\nPrediction: {'Tumor' if prediction == 1 else 'Normal'}"
        color = (0, 1, 0) if label == prediction else (1, 0, 0)  # Green if correct, red if incorrect
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Save the image with the predicted label
        output_path = os.path.join(output_dir, f"image_{i}.jpg")
        cv2.imwrite(output_path, img)

# Path to tumor and normal brain image folders
tumor_folder = r"C:\Users\User\Downloads\DIP project\brain_tumor_dataset\yes"
normal_folder = r"C:\Users\User\Downloads\DIP project\brain_tumor_dataset\no"

# Load tumor images
tumor_images, tumor_labels = load_images(tumor_folder)

# Load normal brain images
normal_images, normal_labels = load_images(normal_folder)

# Combine tumor and normal brain images and labels
images = tumor_images + normal_images
labels = tumor_labels + normal_labels

# Convert images and labels to NumPy arrays
X = np.array(images)
y = np.array(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the pixel values to the range [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape the input data for CNN
X_train = X_train.reshape(-1, 64, 64, 3)
X_test = X_test.reshape(-1, 64, 64, 3)

# Create the CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile and train the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Predict on the test set
y_pred = model.predict(X_test)
y_pred = np.round(y_pred).flatten()

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save the predicted labels to a file
#output_file = "predicted_labels.txt"
#with open(output_file, 'w') as f:
   # for label in y_pred:
      #  f.write(str(label) + '\n')

#print("Predicted labels saved to:", output_file)

# Save and plot the output images with predicted labels
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)
save_predicted_images(X_test * 255, y_test, y_pred, output_dir)

# Plot the output images with highlighting
num_images = min(5, len(os.listdir(output_dir)))
fig, axes = plt.subplots(2, num_images, figsize=(10, 4))
for i in range(num_images):
    img_path = os.path.join(output_dir, f"image_{i}.jpg")
    if os.path.isfile(img_path):
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        label = y_test[i]
        prediction = y_pred[i]

        # Highlight the tumor region in tumor images
        if label == 1:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, (0, 0, 255), 2)

        color = (0, 1, 0) if label == prediction else (1, 0, 0)  # Green if correct, red if incorrect

        axes[0, i].imshow(X_test[i])
        axes[0, i].axis('off')
        axes[0, i].set_title('Actual')
        axes[0, i].text(0, -15, 'Tumor' if label == 1 else 'Normal', color=color)

        axes[1, i].imshow(img)
        axes[1, i].axis('off')
        axes[1, i].set_title('Predicted')
        axes[1, i].text(0, -15, 'Tumor' if prediction == 1 else 'Normal', color=color)

plt.tight_layout()
plt.show()


# In[3]:


import os
import cv2
import matplotlib.pyplot as plt

tumor_folder = r"C:\Users\User\Downloads\DIP project\brain_tumor_dataset\yes"
normal_folder = r"C:\Users\User\Downloads\DIP project\brain_tumor_dataset\no"
output_folder = r"C:\Users\User\Downloads\DIP project\highlighted_images"

# Function to highlight the tumor region in an image
def highlight_tumor(image, label):
    if label == "tumor":
        color = (0, 0, 255)  # Red color for tumor images
    else:
        color = (0, 255, 0)  # Green color for normal images
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to segment the tumor region
    _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours of the tumor region
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on the image to highlight the tumor region
    cv2.drawContours(image, contours, -1, color, 2)
    
    return image

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process images from the tumor folder
for filename in os.listdir(tumor_folder):
    img_path = os.path.join(tumor_folder, filename)
    img = cv2.imread(img_path)
    if img is not None:
        highlighted_img = highlight_tumor(img, "tumor")
        output_path = os.path.join(output_folder, f"tumor_{filename}")
        cv2.imwrite(output_path, highlighted_img)

# Process images from the normal folder
for filename in os.listdir(normal_folder):
    img_path = os.path.join(normal_folder, filename)
    img = cv2.imread(img_path)
    if img is not None:
        highlighted_img = highlight_tumor(img, "normal")
        output_path = os.path.join(output_folder, f"normal_{filename}")
        cv2.imwrite(output_path, highlighted_img)

# Display the highlighted images using matplotlib
fig, axes = plt.subplots(2, 5, figsize=(12, 6))

# Plot tumor images
for i, filename in enumerate(os.listdir(tumor_folder)[:5]):
    img_path = os.path.join(output_folder, f"tumor_{filename}")
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    axes[0, i].imshow(img)
    axes[0, i].axis('off')
    axes[0, i].set_title('Tumor')

# Plot normal images
for i, filename in enumerate(os.listdir(normal_folder)[:5]):
    img_path = os.path.join(output_folder, f"normal_{filename}")
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    axes[1, i].imshow(img)
    axes[1, i].axis('off')
    axes[1, i].set_title('Normal')

plt.tight_layout()
plt.show()


# In[1]:


import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Function to load images from a folder
def load_images(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (224, 224))
            images.append(img)
            labels.append(1 if 'yes' in filename else 0)  # Set label 1 for tumor images, 0 for normal images
    return images, labels

# Function to save images with predicted labels
def save_predicted_images(images, labels, predictions, output_dir):
    for i in range(len(images)):
        img = images[i]
        label = labels[i]
        prediction = predictions[i]

        # Add label and prediction text to the image
        text = f"Label: {'Tumor' if label == 1 else 'Normal'}\nPrediction: {'Tumor' if prediction == 1 else 'Normal'}"
        color = (0, 1, 0) if label == prediction else (1, 0, 0)  # Green if correct, red if incorrect
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Save the image with the predicted label
        output_path = os.path.join(output_dir, f"image_{i}.jpg")
        cv2.imwrite(output_path, img)

# Path to tumor and normal brain image folders
tumor_folder = r"C:\Users\User\Downloads\DIP project\brain_tumor_dataset\yes"
normal_folder = r"C:\Users\User\Downloads\DIP project\brain_tumor_dataset\no"

# Load tumor images
tumor_images, tumor_labels = load_images(tumor_folder)

# Load normal brain images
normal_images, normal_labels = load_images(normal_folder)

# Combine tumor and normal brain images and labels
images = tumor_images + normal_images
labels = tumor_labels + normal_labels

# Convert images and labels to NumPy arrays
X = np.array(images)
y = np.array(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the pixel values to the range [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# Load the VGG16 model without the top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the convolutional base
base_model.trainable = False

# Create a new model by adding the top layers to the frozen base model
model = models.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile and train the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Predict on the test set
y_pred = model.predict(X_test)
y_pred = np.round(y_pred).flatten()

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save the predicted labels to a file
output_file = "predicted_labels.txt"
with open(output_file, 'w') as f:
    for label in y_pred:
        f.write(str(label) + '\n')

print("Predicted labels saved to:", output_file)

# Save and plot the output images with predicted labels
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)
save_predicted_images(X_test * 255, y_test, y_pred, output_dir)

# Plot the output images with highlighting
num_images = min(5, len(os.listdir(output_dir)))
fig, axes = plt.subplots(2, num_images, figsize=(10, 4))
for i in range(num_images):
    img_path = os.path.join(output_dir, f"image_{i}.jpg")
    if os.path.isfile(img_path):
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        label = y_test[i]
        prediction = y_pred[i]

        # Highlight the tumor region in tumor images
        if label == 1:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, (0, 0, 255), 2)

        color = (0, 1, 0) if label == prediction else (1, 0, 0)  # Green if correct, red if incorrect

        axes[0, i].imshow(X_test[i])
        axes[0, i].axis('off')
        axes[0, i].set_title('Actual')
        axes[0, i].text(0, -15, 'Tumor' if label == 1 else 'Normal', color=color)

        axes[1, i].imshow(img)
        axes[1, i].axis('off')
        axes[1, i].set_title('Predicted')
        axes[1, i].text(0, -15, 'Tumor' if prediction == 1 else 'Normal', color=color)

plt.tight_layout()
plt.show()


# In[1]:


def chat_bot():
    print("Bot: Why do you love me?")
    response = input("You: ")
    
    if "care for others" in response:
        print("Bot: Because you care for others.")
        print("Bot: That's one of the many reasons why I love you.")
    elif "inspire me" in response:
        print("Bot: Because you inspire me.")
        print("Bot: Your strength and determination motivate me every day.")
    elif "smile when you're nervous" in response:
        print("Bot: Because the way you smile when you're nervous.")
        print("Bot: It shows your vulnerability and authenticity, and I find it endearing.")
    elif "beautiful in every way" in response:
        print("Bot: Because you're beautiful in every way.")
        print("Bot: Your inner and outer beauty radiates, and I'm in awe of you.")
    elif "everything" in response:
        print("Bot: Sweetheart, you're not just a 'that's it', you're my everything.")
        print("Bot: Your presence in my life brings me immense joy and love.")
    else:
        print("Bot: I'm sorry, I didn't understand your response. Can you please clarify?")
        chat_bot()  # Recursive call to restart the conversation

# Run the chat bot
chat_bot()


# In[2]:


def chat_bot():
    responses = [
        "Because you care for others",
        "Because you inspire me",
        "Because the way you smile when you're nervous",
        "Because you're beautiful in every way",
        "Sweetheart, you're not just a 'that's it', you're my everything"
    ]
    
    print("Me: Why do you love me?")
    
    for response in responses:
        print("System:", response)
        user_response = input("Me: ")
        
        if user_response.lower() == "that's it?":
            continue
        
        if user_response.lower() == "sweetheart you're not a that's it, you're my everything":
            print("System: That's right! You mean everything to me.")
            break
        
        print("System: I'm sorry, I didn't understand your response. Can you please clarify?")
    
    print("Conversation ended.")

# Run the chat bot
chat_bot()


# In[ ]:


def chat_bot():
    print("Hello! I'm your chatbot. How can I assist you today?")
    
    while True:
        user_input = input("You: ")
        response = get_bot_response(user_input)
        print("Bot:", response)
        
        if user_input.lower() == "bye":
            print("Bot: Goodbye! Have a great day!")
            break

def get_bot_response(user_input):
    # Define bot responses based on user input
    if "hello" in user_input.lower() or "hi" in user_input.lower():
        return "Hi there! How can I help you?"
    elif "how are you" in user_input.lower():
        return "I'm a bot, so I don't have feelings, but I'm here to assist you!"
    elif "thank you" in user_input.lower() or "thanks" in user_input.lower():
        return "You're welcome! I'm here to help anytime."
     elif "fuck you" in user_input.lower() or "fuck youuuuuuu " in user_input.lower():
        return "Anytime."
    else:
        return "I'm sorry, I didn't understand. Can you please rephrase or ask a different question?"

# Run the chat bot
chat_bot()


# In[1]:


import cv2
import numpy as np
import os

def region_growing(image, seed):
    # Create a mask of the same size as the image, initialized with zeros
    height, width = image.shape[:2]
    segmented = np.zeros_like(image, dtype=np.uint8)
    
    # Define the connectivity (8-connectivity)
    connectivity = [(0, 1), (1, 0), (-1, 0), (0, -1), (1, 1), (-1, -1), (-1, 1), (1, -1)]
    
    # Set the seed point as the initial region
    region_points = [seed]
    segmented[seed] = image[seed]
    
    # Start region growing
    while len(region_points) > 0:
        # Get the last point in the region
        current_point = region_points[-1]
        region_points.pop()
        
        # Check the neighbors of the current point
        for dx, dy in connectivity:
            x = current_point[0] + dx
            y = current_point[1] + dy
            
            # Check if the neighbor is within the image bounds
            if x >= 0 and x < height and y >= 0 and y < width:
                # Check if the neighbor is unlabeled and similar to the current region
                if segmented[x, y] == 0 and abs(int(image[x, y]) - int(image[current_point])) <= 10:
                    segmented[x, y] = image[x, y]
                    region_points.append((x, y))
    
    return segmented

# Path to the tumor and normal brain image folders
tumor_folder = r"C:\Users\User\Downloads\DIP project\brain_tumor_dataset\yes"
normal_folder = r"C:\Users\User\Downloads\DIP project\brain_tumor_dataset\no"

# Threshold area to classify tumors
threshold_area = 5000  # Adjust this value based on your specific requirements

# Process tumor images
for filename in os.listdir(tumor_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Load the brain MRI image
        image = cv2.imread(os.path.join(tumor_folder, filename), 0)  # Load as grayscale

        # Apply Canny edge detection
        edges = cv2.Canny(image, 100, 200)

        # Find contours of the detected edges
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Iterate through the contours
        for contour in contours:
            # Calculate the area and perimeter of the contour
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            # Skip small contours (noise)
            if area < 100:
                continue

            # Calculate the centroid of the contour
            moments = cv2.moments(contour)
            centroid_x = int(moments['m10'] / moments['m00'])
            centroid_y = int(moments['m01'] / moments['m00'])
            seed_point = (centroid_y, centroid_x)  # Swap x and y due to numpy indexing

            # Apply region growing to segment the tumor region
            segmented_tumor = region_growing(image, seed_point)

            # Perform classification based on area
            if area > threshold_area:
                tumor_class = 'Malignant'
            else:
                tumor_class = 'Benign'

            # Display the original image, Canny edges, segmented tumor, and classification result
            cv2.imshow('Original Image', image)
            cv2.imshow('Canny Edges', edges)
            cv2.imshow('Segmented Tumor', segmented_tumor)
            cv2.putText(segmented_tumor, tumor_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Classification Result', segmented_tumor)
            cv2.waitKey(0)

# Process normal images
for filename in os.listdir(normal_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Load the brain MRI image
        image = cv2.imread(os.path.join(normal_folder, filename), 0)  # Load as grayscale

        # Apply Canny edge detection
        edges = cv2.Canny(image, 100, 200)

        # Find contours of the detected edges
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Iterate through the contours
        for contour in contours:
            # Calculate the area and perimeter of the contour
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            # Skip small contours (noise)
            if area < 100:
                continue

            # Calculate the centroid of the contour
            moments = cv2.moments(contour)
            centroid_x = int(moments['m10'] / moments['m00'])
            centroid_y = int(moments['m01'] / moments['m00'])
            seed_point = (centroid_y, centroid_x)  # Swap x and y due to numpy indexing

            # Apply region growing to segment the tumor region
            segmented_tumor = region_growing(image, seed_point)

            # Perform classification based on area
            if area > threshold_area:
                tumor_class = 'Malignant'
            else:
                tumor_class = 'Benign'

            # Display the original image, Canny edges, segmented tumor, and classification result
            cv2.imshow('Original Image', image)
            cv2.imshow('Canny Edges', edges)
            cv2.imshow('Segmented Tumor', segmented_tumor)
            cv2.putText(segmented_tumor, tumor_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Classification Result', segmented_tumor)
            cv2.waitKey(0)

cv2.destroyAllWindows()


# In[ ]:


import cv2
import numpy as np
import os

def region_growing(image, seed):
    # Create a mask of the same size as the image, initialized with zeros
    height, width = image.shape[:2]
    segmented = np.zeros((height, width), dtype=np.uint8)

    # Define the connectivity (8-connectivity)
    connectivity = [(0, 1), (1, 0), (-1, 0), (0, -1), (1, 1), (-1, -1), (-1, 1), (1, -1)]

    # Set the seed point as the initial region
    region_points = [seed]
    segmented[seed] = image[seed]

    # Start region growing
    while len(region_points) > 0:
        # Get the last point in the region
        current_point = region_points.pop()

        # Check the neighbors of the current point
        for dx, dy in connectivity:
            x = current_point[0] + dx
            y = current_point[1] + dy

            # Check if the neighbor is within the image bounds
            if 0 <= x < height and 0 <= y < width:
                # Check if the neighbor is unlabeled and similar to the current region
                if segmented[x, y] == 0 and abs(int(image[x, y]) - int(image[current_point])) <= 10:
                    segmented[x, y] = image[x, y]
                    region_points.append((x, y))

    return segmented


# Path to the tumor and normal brain image folders
tumor_folder = r"C:\Users\User\Downloads\DIP project\brain_tumor_dataset\yes"
normal_folder = r"C:\Users\User\Downloads\DIP project\brain_tumor_dataset\no"

# Threshold area to classify tumors
threshold_area = 5000  # Adjust this value based on your specific requirements

# Process tumor images
for filename in os.listdir(tumor_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Load the brain MRI image
        image = cv2.imread(os.path.join(tumor_folder, filename), cv2.IMREAD_GRAYSCALE)

        # Apply Canny edge detection
        edges = cv2.Canny(image, 100, 200)

        # Find contours of the detected edges
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Iterate through the contours
        for contour in contours:
            # Calculate the area and perimeter of the contour
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            # Skip small contours (noise)
            if area < 100:
                continue

            # Calculate the centroid of the contour
            moments = cv2.moments(contour)
            centroid_x = int(moments['m10'] / moments['m00'])
            centroid_y = int(moments['m01'] / moments['m00'])
            seed_point = (centroid_x, centroid_y)  # Swap x and y due to numpy indexing

            # Apply region growing to segment the tumor region
            segmented_tumor = region_growing(image, seed_point)

            # Perform classification based on area
            if area > threshold_area:
                tumor_class = 'Malignant'
            else:
                tumor_class = 'Benign'

            # Display the original image, Canny edges, segmented tumor, and classification result
            cv2.imshow('Original Image', image)
            cv2.imshow('Canny Edges', edges)
            cv2.imshow('Segmented Tumor', segmented_tumor)
            cv2.putText(segmented_tumor, tumor_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Classification Result', segmented_tumor)
            cv2.waitKey(0)

# Process normal images
for filename in os.listdir(normal_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Load the brain MRI image
        image = cv2.imread(os.path.join(normal_folder, filename), cv2.IMREAD_GRAYSCALE)

        # Apply Canny edge detection
        edges = cv2.Canny(image, 100, 200)

        # Find contours of the detected edges
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Iterate through the contours
        for contour in contours:
            # Calculate the area and perimeter of the contour
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            # Skip small contours (noise)
            if area < 100:
                continue

            # Calculate the centroid of the contour
            moments = cv2.moments(contour)
            centroid_x = int(moments['m10'] / moments['m00'])
            centroid_y = int(moments['m01'] / moments['m00'])
            seed_point = (centroid_x, centroid_y)  # Swap x and y due to numpy indexing

            # Apply region growing to segment the tumor region
            segmented_tumor = region_growing(image, seed_point)

            # Perform classification based on area
            if area > threshold_area:
                tumor_class = 'Malignant'
            else:
                tumor_class = 'Benign'

            # Display the original image, Canny edges, segmented tumor, and classification result
            cv2.imshow('Original Image', image)
            cv2.imshow('Canny Edges', edges)
            cv2.imshow('Segmented Tumor', segmented_tumor)
            cv2.putText(segmented_tumor, tumor_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Classification Result', segmented_tumor)
            cv2.waitKey(0)

cv2.destroyAllWindows()


# In[ ]:




