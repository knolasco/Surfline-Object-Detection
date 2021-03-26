![Clip_for_LinkedIn2](https://user-images.githubusercontent.com/66560796/111404254-af074f00-868b-11eb-8447-4631976df973.gif)

# Introduction

Surfline is a tech company based in Huntington Beach, CA. Surfline offers a variety of services for surf forecasting. The purpose of this project is to take the existing surfline cameras and make them intelligent. For this project, we are using transfer learning to train a model to detect different directions of waves. When surfing, a surfer can take-off at the peak of a wave and carve to the left or right. By using computer vision, we could try to provide surfers with a summary of how the waves are behaving at a particular time. Some surfers prefer lefts, while others prefer rights and this summary could help the surfers have clear expectations.

# Description of Files

### Centroid_Tracker.py
This python script utilizes a Centroid Tracking algorithm to record the number of waves and the average length of each wave. 

### Requirements.txt
This text file defines the packages used for this project.

### Wave_Detection.ipynb
This is a Google Colab notebook that I used to perform transfer learning. Since I used a deep neural network for this project, I wanted to use the free GPU that is offered by Google Colab to speed up the processing time.

### Wave_Detection.py
This python script loads the model from the Google Colab notebook and uses OpenCV to detect waves in real time.

### Preprocess_Images.ipynb
This is a Google Colab notebook where I experimented with preprocessing the train/test images. The model that was used for the GIF above did not use these preprocessed images, but this technique could be used to improve the accuracy of the model.

### Flip_and_save.py
This python script was used to flip all of the train/test images. The images that were used to train the model contained more "rights" than "lefts" which could make the model biased towards rights. The reason there were more rights is because Southern California currently has a NW swell, which causes waves to break towards the right more often than to the left. The model could be trained again to include the original images and the flipped images to avoid the bias and make the model generalize better.
