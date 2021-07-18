# import necessary packages
from pyimagesearch.utils import get_class_idx
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import decode_predictions
from tensorflow.keras.applications.resnet50 import preprocess_input
import argparse
import imutils
import numpy as np
import cv2

def preprocess_image(image):
	# swap color channels, preprocess the image, and add in a batch
	# dimension
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = preprocess_input(image)
	image = cv2.resize(image, (224, 224))
	image = np.expand_dims(image, axis=0)
	# return the preprocessed image
	return image

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to input image")
args = vars(ap.parse_args())

print("[INFO] loading image...")
image = cv2.imread(args["image"])
output = image.copy()
    
# preprocess the input image
output = imutils.resize(output, width=400)
preprocessedImage = preprocess_image(image)


# load the pre-trained ResNet50 model
print("[INFO] loading pre-trained ResNet50 model...")
model = ResNet50(weights="imagenet")
    
# make predictions on the input image and parse the top-3 predictions
print("[INFO] making predictions...")
predictions = model.predict(preprocessedImage)
predictions = decode_predictions(predictions, top=3)[0]

# loop over the top three predictions
for (i, (imagenetID, label, prob)) in enumerate(predictions):
    # print the ImageNet class label ID of the top prediction to our
    # terminal (we'll need this label for our next script which will
    # perform the actual adversarial attack)
    if i == 0:
        print("[INFO] {} => {}".format(label, get_class_idx(label)))

    # display the prediction to our screen
    print("[INFO] {}. {}: {:.2f}%".format(i + 1, label, prob * 100))

# draw the top-most predicted label on the image along with the
# confidence score
text = "{}: {:.2f}%".format(predictions[0][1],
    predictions[0][2] * 100)
cv2.putText(output, text, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
    (0, 255, 0), 2)
# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)