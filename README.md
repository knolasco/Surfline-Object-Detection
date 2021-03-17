![Clip_for_LinkedIn2](https://user-images.githubusercontent.com/66560796/111404254-af074f00-868b-11eb-8447-4631976df973.gif)

# Introduction

Surfline is a tech company based in Huntington Beach, CA. Surfline offers a variety of services for surf forecasting. The purpose of this project is to take the existing surfline cameras and make them intelligent. For this project, we are using transfer learning to train a model to detect different directions of waves. When surfing, a surfer can take-off at the peak of a wave and carve to the left or right. By using computer vision, we could try to provide surfers with a summary of how the waves are behaving at a particular time. Some surfers prefer lefts, while others prefer rights and this summary could help the surfers have clear expectations.

# Description of Files

### Requirements.txt
This text file defines the packages used for this project.

### Wave_Detection.ipynb
This is a Google Colab notebook that I used to perform transfer learning. Since I used a deep neural network for this project, I wanted to use the free GPU that is offered by Google Colab to speed up the processing time.

### Wave_Detection.py
This python script loads the model from the Google Colab notebook and uses OpenCV to detect waves in real time.
