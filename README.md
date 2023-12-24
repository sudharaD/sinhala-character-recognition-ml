# Sinhala Character Recognition using KNN Algorithm

This project aims to recognize Sinhala characters using the K-Nearest Neighbors (KNN) algorithm. The project includes a graphical user interface (GUI) for drawing Sinhala characters, a dataset creation part, model training, and a GUI for character recognition.

### Dataset Creation

The dataset creation part involves organizing Sinhala character images and creating data and target arrays for training the machine learning model. Images are loaded, resized, and converted to grayscale before being used to train the KNN algorithm.

## Dataset Creation

The dataset creation part involves organizing Sinhala character images and creating data and target arrays for training the machine learning model. Images are loaded, resized, and converted to grayscale before being used to train the KNN algorithm.

```python
# Code for creating the dataset

# Create the category dictionary
def create_category_dict(data_path):
    categories = os.listdir(data_path)
    labels = [i for i in range(len(categories))]
    category_dict = {}
    for i in range(len(categories)):
        category_dict[categories[i]] = labels[i]
    return category_dict, categories

data_path = 'dataset'
category_dict, categories = create_category_dict(data_path)

# ... (code for displaying images)

# Create data & target data sets
def create_data_target(data, target):
    for category in categories:
        imgs_path = os.path.join(data_path, category)
        img_names = os.listdir(imgs_path)
        for img_name in img_names:
            img_path = os.path.join(imgs_path, img_name)
            img = cv2.imread(img_path, 0)
            img = cv2.resize(img, (8,8))
            data.append(img)
            target.append(category_dict[category])

    return data, target

data = []
target = []
data, target = create_data_target(data, target)

# Save data and target arrays to .npy files
np.save('data/data', data)
np.save('data/target', target)
```

### Model Training

The model training part loads the saved data and target arrays, splits them into training and testing sets, and trains the KNN classifier. The accuracy and classification report of the model are displayed, and the trained model is saved for later use.

```python
# Code for training the KNN model
import numpy as np
import joblib

# Load the saved data & target
data = np.load('data/data.npy')
target = np.load('data/target.npy')

# Split the dataset to training & testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

# Load the KNN algorithm
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()

# Train the model
model.fit(X_train, y_train)

# Get predictions
y_predict = model.predict(X_test)

# Model Accuracy
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_predict)

# Classification Report
from sklearn.metrics import classification_report
cla_report = classification_report(y_test, y_predict)

# Save the model
joblib.dump(model, 'data/sinhala-character-KNN.sav')

```

### GUI for Sinhala Character Input

The graphical user interface allows users to draw Sinhala characters using a canvas. The drawn character can be saved, cleared, or predicted using the trained KNN model. The predicted character is displayed on the GUI.

```python
win = tk.Tk()

model = joblib.load('data/sinhala-character-KNN.sav')

width = 500
height = 500
font_btn = 'Helvetica 20 bold'
font_lbl = 'Helvetica 25 bold'
count = 0
label_dict = {0:'අ', 1:' එ', 2:' ඉ', 3:'උ'}

# Letter drawing function
def event_function(event):
    x = event.x
    y = event.y

    x1, x2 = x-30, x+30
    y1, y2 = y-30, y+30

    # Draw the oval when curser moving
    canvas.create_oval((x1, y1, x2, y2), fill='black')

    img_draw.ellipse((x1, y1, x2, y2), fill='black')

def save():

    global count

    img_array = np.array(img)
    path = os.path.join('data/save/', str(count)+'.jpg')
    # print(path)
    # img_array = cv2.resize(img_array, (8,8))
    cv2.imwrite(path, img_array)
    count += 1

def clear():

    global img, img_draw

    canvas.delete('all')
    img = Image.new('RGB', (width, height), (255,255,255))
    img_draw = ImageDraw.Draw(img)

    lbl_status.config(text='PREDICTED CHARACTER: NONE')

def predict():

    img_array = np.array(img)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY) # Converting into a gray image
    img_array = cv2.resize(img_array, (8,8))
    img_array = np.reshape(img_array, (1, 64))

    result = model.predict(img_array)[0]

    label = label_dict[result]

    lbl_status.config(text='PREDICTED CHARACTER: '+label)

# Create a canvas and position it within the window
canvas = tk.Canvas(win, width=width, height=height, bg='white')
canvas.grid(row=0, column=0, columnspan=4)

# Creates the buttons and position them within the window
btn_save = tk.Button(win, text='SAVE', bg='green', fg='white', font=font_btn, command=save)
btn_save.grid(row=1, column=0)

btn_save = tk.Button(win, text='PREDICT', bg='blue', fg='white', font=font_btn, command=predict)
btn_save.grid(row=1, column=1)

btn_save = tk.Button(win, text='CLEAR', bg='yellow', fg='white', font=font_btn, command=clear)
btn_save.grid(row=1, column=2)

btn_save = tk.Button(win, text='EXIT', bg='red', fg='white', font=font_btn, command=win.destroy)
btn_save.grid(row=1, column=3)

# Create a label and position it within the window
lbl_status = tk.Label(win, text='PREDICTED CHARACTER: NONE', bg='white', font=font_lbl)
lbl_status.grid(row=2, column=0, columnspan=4)

# Bind the event with the canvas
canvas.bind('<B1-Motion>', event_function)

img = Image.new('RGB', (width, height), (255,255,255))
img_draw = ImageDraw.Draw(img)

win.mainloop()
```

### Instructions for Running the GUI:

1. Run the dataset creation code to organize and save the Sinhala character images.
2. Execute the model training code to train the KNN algorithm.
3. Run the GUI creation code to open the interface for drawing Sinhala characters.

### GUI Controls:

- **SAVE:** Save the drawn character for future training.
- **PREDICT:** Use the trained model to predict the drawn character.
- **CLEAR:** Clear the canvas for a new drawing.
- **EXIT:** Close the GUI.

### Note:

- Ensure that the required libraries (OpenCV, NumPy, Matplotlib, Pillow) are installed before running the code.
- Make sure to run the steps in order to create the dataset, train the model, and then use the GUI.
