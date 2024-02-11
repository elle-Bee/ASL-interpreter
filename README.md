# ASL interpreter

https://github.com/elle-Bee/ASL-interpreter/assets/65856801/840d116a-2f71-42e0-ae65-bc3f8670f465

## Requirements
- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Mediapipe
- Scikit-learn
- Flask

## Words availabe for interpretation
| Letter | Words                                     |
|--------|-------------------------------------------|
| A      | a, again, applause                        |
| B      | b, baby, boy                              |
| C      | c, cold, cookie, cup                      |
| D      | d, dad, drink                             |
| E      | e                                         |
| F      | f, food, full                             |
| G      | g, girl                                   |
| H      | h, help                                   |
| I      | i                                         |
| K      | k, know                                   |
| L      | l, learn, love                            |
| M      | m, mad, marriage, mom, money, more       |
| N      | n, name, no                               |
| O      | o                                         |
| P      | p, person, please                         |
| Q      | q                                         |
| R      | r                                         |
| S      | s, sad, shirt, short, sorry, stop         |
| T      | t, tall, teach, thank you                 |
| U      | u, understand                            |
| V      | v                                         |
| W      | w, water, what                            |
| X      | x                                         |
| Y      | y, yellow, yes                            |
| Z      | z                                         |
| Numbers   | 3, 5, 6, 7, 8, 9, 10                       |


## Usage
1. Clone the repository.
2. Make sure you have Python 3.x installed on your system along with the required libraries.
```shell
  python -m pip install -r requirements.txt
```
3. Go into the root folder of the project and run the app by using the following command.
```shell
   python app.py
```
4. Go to the development server at: `http://localhost:5000`

## Project Structure

- `app.py`: Main Flask application script.
- `templates/index.html`: HTML template for rendering the web page.
- `inference_classifier.py`: Module containing the GestureClassifier class for performing gesture recognition.

## Navigating through the project

### `scripts/01_collect_imgs.py`

This script allows you to collect real-time image data from your webcam with specified labels. It creates a dataset for each label by capturing images and storing them in separate directories within a specified data directory.<br>

#### Usage

1. Run the script.
2. Enter the labels you want to create when prompted. Enter `-1` to stop adding labels.
3. Once labels are entered, the webcam will activate.
4. Press `Q` to start capturing images for each label.
5. Images will be stored in the specified data directory under separate folders for each label.

#### Parameters

- `DATA_DIR`: Directory to store the collected data. Default is `./data`.
- `dataset_size`: Number of images to collect for each label. Default is `300`.

#### Notes

- Ensure proper lighting and background for accurate image collection.
- Press `Q` to start capturing images after each label prompt.
---

### `scripts/02_create_dataset.py`

This script captures images from a specified directory, detects hand landmarks using the MediaPipe library, and saves the landmark data along with corresponding labels into a pickle file.

#### Usage

1. Place your image data in the specified data directory (`./data` by default).
2. Run the script.
3. The script will process each image, extract hand landmarks, and save the data along with labels into a pickle file named `data.pickle`.

#### Parameters

- `DATA_DIR`: Directory containing the image data. Default is `./data`.

#### Notes

- Ensure your images have sufficient resolution and quality for accurate hand landmark detection.
- The script assumes that each subdirectory in the data directory represents a different label/class.
- Hand landmark data is saved as a list of coordinates relative to the top-left corner of the bounding box of the detected hand.
- The pickle file `data.pickle` contains a dictionary with keys 'data' and 'labels', where 'data' is a list of hand landmark data and 'labels' is a list of corresponding labels.
---

### `scripts/03_train_classifier.py`

This script trains a Random Forest classifier for gesture recognition using hand landmarks data. It also evaluates the model's performance using cross-validation and saves the trained model for future use.

#### Usage

1. Ensure you have hand landmarks data saved as `data.pickle` in the project directory.
2. Run the script.
3. The script will load the hand landmarks data, preprocess it, train a Random Forest classifier, and evaluate its performance.

#### Notes

- Hand landmarks data should be saved as a dictionary (`labels_dict.py`)containing 'data' (list of hand landmark data) and 'labels' (list of corresponding labels).
- The script pads each hand landmark sequence with zeros to ensure all sequences have the same length, necessary for training the classifier.
- The classifier is trained using stratified train-test split and evaluated using cross-validation for robustness.
- The trained model is saved as `model.p` using the `pickle` module for future use.
- Adjust the model parameters and preprocessing steps as needed for improved performance.
---

### `scripts/04_inference_classifier.py`

This script performs real-time gesture recognition using hand landmarks detected by the MediaPipe library. It loads a pre-trained gesture classification model and overlays the predicted gesture label on the input video stream.

#### Usage

1. Ensure you have a trained gesture classification model saved as `model.p` in the project directory.
2. Run the script.
3. The script will activate your webcam and overlay the predicted gesture label on the detected hand landmarks in real-time.

#### Notes

- The gesture classification model is assumed to be trained externally and saved using the `pickle` module.
- Hand landmarks are detected using the MediaPipe library, providing a robust representation of hand gestures.
- The script draws bounding boxes around detected hands and overlays the predicted gesture label on the video stream.
- Adjust the `min_detection_confidence` parameter of the `Hands` class for controlling the confidence threshold of hand landmark detection.
- Ensure proper lighting and background for accurate hand landmark detection and gesture recognition.
---

### `app.py`

This Flask-based web application streams real-time video from your webcam and performs gesture recognition using a pre-trained model. The predicted gesture labels are overlaid on the video stream and displayed on a web page.

#### Usage

1. Ensure you have your pre-trained gesture classification model saved and inference code ready.
2. Run the Flask application (`app.py`).
3. Open your web browser and navigate to `http://localhost:5000` or `http://127.0.0.1:5000`.
4. You should see the real-time video stream with predicted gesture labels overlaid.

#### Notes

- The `GestureClassifier` class is assumed to be implemented in `inference_classifier.py`.
- The Flask application captures frames from the webcam using OpenCV, performs gesture recognition using the `GestureClassifier` class, and streams the processed frames to the web page.
- Ensure proper permissions for accessing the webcam.
- Adjust the URL (`http://localhost:5000`) according to your Flask application settings.
---
