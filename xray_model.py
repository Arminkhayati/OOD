import os
import tensorflow as tf
import numpy as np
from PIL import Image
from oodeel.methods import Entropy, Energy, MLS

IMG_SIZE=224
TARGET_SIZE = (IMG_SIZE, IMG_SIZE)

class XrayModel:

    def __init__(self,
                 model_path="model/vgg16-0.96-full_model.h5",
                 ood_method=None,
                 data_dir="data/x_ray",):
        self.data_dir = data_dir
        self.model = tf.keras.models.load_model(model_path)
        self.ood_method = ood_method
        self.threshold = 0
        self.methods_list = [
            None,
            "MSP",
            "MLS",
            "Entropy",
            "Energy"
        ]
        self.label_decoding = {0: "Dandan", 1: "JomJome", 2: "Sine", 3: "Dast", 4: "Pa", 5: "Unknown"}
        if self.ood_method != None:
            self.__load_odd_metric()



    def __load_and_process(self, img):
        img = Image.open(os.path.join(self.data_dir, img))
        img = img.convert('L')
        img = img.convert('RGB')
        img = img.resize(TARGET_SIZE)
        img = np.array(img)
        img = img/255.0
        img = np.expand_dims(img, axis=0)
        return img

    def predict(self, img_file):
        img = self.__load_and_process(img_file)
        if self.ood_method != None:
            predicted_class_indices = self.__get_class(img)
        else:
            preds = self.model.predict(img, verbose=0)
            predicted_class_indices = np.argmax(preds, axis=1)
        predicted_classes = [self.label_decoding[l] for l in predicted_class_indices]
        return predicted_classes
        # return predicted_class_indices[0]


    def __get_class(self, img):
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


    def __load_odd_metric(self):
        if self.ood_method == "MLS":
            self.ood_finder = MLS()
            self.ood_finder.fit(self.model)
            self.threshold = -7.625665473937988
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
            self.threshold = -7.628254842758179
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





