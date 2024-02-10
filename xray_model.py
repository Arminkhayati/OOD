import os
import tensorflow as tf
import numpy as np
from PIL import Image
from oodeel.methods import Entropy, Energy, MLS
from typing import List, Optional, Union

IMG_SIZE=224
TARGET_SIZE = (IMG_SIZE, IMG_SIZE)

class XrayModel:
    """A class to handle X-ray image predictions with optional out-of-distribution detection.

        Attributes:
            data_dir (str): Directory where X-ray images are stored.
            model (tf.keras.Model): Loaded TensorFlow model for predictions.
            ood_method (Optional[str]): Method used for out-of-distribution detection.
            threshold (float): Threshold value for out-of-distribution detection.
            methods_list (List[str]): List of supported methods for out-of-distribution detection.
            label_decoding (dict): Mapping of class indices to human-readable labels.
            return_label (bool): Whether to return string label or int index of the predicted class.
        """

    def __init__(self,
                 model_path: str = "model/vgg16-0.96-full_model.h5",
                 ood_method: Optional[str] = None,
                 data_dir: str = "data/x_ray",
                 strict: bool = True,
                 return_label: bool = False):
        """Initializes the XrayModel with a model path, out-of-distribution method, and data directory."""
        self.data_dir: str = data_dir
        self.model: tf.keras.Model = tf.keras.models.load_model(model_path)
        self.ood_method: Optional[str] = ood_method
        self.threshold: float  = 0
        self.methods_list: List[str] = [
            None,
            "MSP",
            "MLS",
            "Entropy",
            "Energy"
        ]
        self.label_decoding: dict = {0: "Dandan", 1: "JomJome", 2: "Sine", 3: "Dast", 4: "Pa", 5: "Unknown"}
        self.return_label = return_label
        if self.ood_method is not None:
            if strict:
                self.__load_odd_metric_strict()
            else:
                self.__load_odd_metric()



    def __load_and_process(self, img: str) -> np.ndarray:
        """Loads and preprocesses an image file for prediction.

                Args:
                    img (str): Filename of the image to load and process.

                Returns:
                    np.ndarray: Preprocessed image ready for model prediction.
        """
        img = Image.open(os.path.join(self.data_dir, img))
        img = img.convert('L')
        img = img.convert('RGB')
        img = img.resize(TARGET_SIZE)
        img = np.array(img)
        img = img/255.0
        img = np.expand_dims(img, axis=0)
        return img

    def predict(self, img_file: str) -> List[str]:
        """Predicts the class of an X-ray image.

        Args:
            img_file (str): Filename of the image to predict.

        Returns:
            List[str]: Predicted class label(s) of the input image.
        """
        img = self.__load_and_process(img_file)
        if self.ood_method is not None:
            predicted_class_indices = self.__get_class(img)
        else:
            preds = self.model.predict(img, verbose=0)
            predicted_class_indices = np.argmax(preds, axis=1)

        if self.return_label:
            predicted_classes = [self.label_decoding[l] for l in predicted_class_indices]
            return predicted_classes
        else:
            return predicted_class_indices
        # return predicted_class_indices[0]


    def __get_class(self, img: np.ndarray) -> List[int]:
        """Determines the class of the image using out-of-distribution detection if enabled.

        Args:
            img (np.ndarray): Preprocessed image for which to determine the class.

        Returns:
            List[int]: Indices of the predicted classes.
        """
        scores, info = self.ood_finder.score(img)
        logits = info['logits']
        predicted_class_indices = []
        for i, score in enumerate(scores):
            if score > self.threshold:
                predicted_class_indices.append(5)
            else:
                sum_logits = sum(logits[i])
                if (sum_logits < 1.1) and (sum_logits > 0.99):
                    logits[i] = tf.math.softmax(logits[i])
                predicted_class_indices.append(np.argmax(logits[i]))

        return predicted_class_indices


    def __load_odd_metric_strict(self):
        """Loads the out-of-distribution detection method based on the specified `ood_method` and strict thresholds."""
        if self.ood_method == "MLS":
            self.ood_finder = MLS()
            self.ood_finder.fit(self.model)
            self.threshold = -7.625577330589294
        elif self.ood_method == "MSP":
            self.ood_finder = MLS(output_activation="softmax")
            self.ood_finder.fit(self.model)
            self.threshold = -0.9606829807162285 #-0.9959281772375107 # -0.9606829807162285
        elif self.ood_method == "Entropy":
            self.ood_finder = Entropy()
            self.ood_finder.fit(self.model)
            self.threshold = 0.027844607923179866
        elif self.ood_method == "Energy":
            self.ood_finder = Energy()
            self.ood_finder.fit(self.model)
            self.threshold = -7.628254842758179 # -6.0133957862854
        else:
            raise ValueError(f"Must be one of these: {self.methods_list}.")

    def __load_odd_metric(self):
        """Loads the out-of-distribution detection method based on the specified `ood_method` and non strict thresholds."""
        if self.ood_method == "MLS":
            self.ood_finder = MLS()
            self.ood_finder.fit(self.model)
            self.threshold = -6.0036115646362305
        elif self.ood_method == "MSP":
            self.ood_finder = MLS(output_activation="softmax")
            self.ood_finder.fit(self.model)
            self.threshold = -0.9902637004852295
        elif self.ood_method == "Entropy":
            self.ood_finder = Entropy()
            self.ood_finder.fit(self.model)
            self.threshold = 0.06262687593698502
        elif self.ood_method == "Energy":
            self.ood_finder = Energy()
            self.ood_finder.fit(self.model)
            self.threshold = -6.0133957862854
        else:
            raise ValueError(f"Must be one of these: {self.methods_list}.")

# methods_list = [
#             "MLS",
#             "MSP",
#             "Entropy",
#             "Energy"
#         ]
#
# def load_and_process(img):
#         img = Image.open(os.path.join(DATA_DIR, img))
#         img = img.convert('L')
#         img = img.convert('RGB')
#         img = img.resize(TARGET_SIZE)
#         img = np.array(img)
#         img = img/255.0
#         img = np.expand_dims(img, axis=0)
#         return img
#
# img = '10030000024.jpg'
# model = tf.keras.models.load_model("model/vgg16-0.96-full_model.h5")
# preds = model.predict(load_and_process(img), verbose=0)
# predicted_class_indices = np.argmax(preds, axis=1)
# print(predicted_class_indices, max(preds[0]))
#
# for m in methods_list:
#     model = XrayModel(ood_method=m)
#     label = model.predict(img)
#     print(label)
# #
#





