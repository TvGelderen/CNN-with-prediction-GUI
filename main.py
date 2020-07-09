import tensorflow as tf
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np

# Load in the test data
(_, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_test_reshaped = x_test.reshape(x_test.shape[0], 28, 28, 1)
# Load in the trained model
model = tf.keras.models.load_model('model/')

# Show images with predictions
index = 0
# Turn all images into Images and then into PhotoImages and put these in an array
def get_images():
    arr = []
    for i in range(1000):
        img = Image.fromarray(x_test[i]).resize((280, 280), resample=Image.NEAREST)
        arr.append(ImageTk.PhotoImage(img))
    return arr
    
def get_predictions():
    # Get softmax output
    arr = model.predict(x_test_reshaped)   
    # Get predicted number
    arr2 = np.array([np.argmax(out) for out in arr])
    # Return both
    return arr, arr2

# When the button is clicked we want to update the index and the image + prediction    
def on_click():
    global index
    index = slider.get()-1
    image_label.config(image=images[index])
    prediction_values.config(text=predictions_values[index])
    prediction_label.config(text=predictions[index])

# Create a windo of 1080 by 720 pixels
window = tk.Tk()
window.geometry('1050x650')
# Grab the images in the required form
images = get_images()
# Get the respective predictions from the model
predictions_values, predictions = get_predictions()

# Create a slider for selecting an index
label1 = tk.Label(text="Use the slider to select an index")
label1.pack()
label1.config(font=("Courier", 16))

slider = tk.Scale(window, from_=1, to=1000, length=1000, orient='horizontal')
slider.pack()

button = tk.Button(window, text="Get index", command=on_click)
button.pack()
button.config(font=("Courier", 12))

image_label = tk.Label(window, image=images[0])
image_label.pack(expand='yes')

label2 = tk.Label(window, text="The model's prediction:")
label2.pack()
label2.config(font=("Courier", 20))

prediction_values = tk.Label(window, text=predictions_values[0])
prediction_values.pack(expand='yes')
prediction_values.config(font=("Courier", 14))

prediction_label = tk.Label(window, text=predictions[0])
prediction_label.pack(expand='yes')
prediction_label.config(font=("Courier", 44))

window.mainloop()