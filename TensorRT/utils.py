from library import *

model_cls_bottle = load_model("model_set/model_cls_bottle_ResNet18.h5")
classes_cls_bottle = { 0:'GOOD',
                      1:'ERROR',}

def load_image_cls_bottle(img):
    img = cv2.resize(img,(90,270))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255   
    # imshow expects values in the range [0, 1]
    return img_tensor

def predict_cnn_bottle(img):
    img = load_image_cls_bottle(img)
    pred_bottle = model_cls_bottle.predict(img,verbose = False)
    value = classes_cls_bottle[np.argmax(pred_bottle)]
    return value, max(pred_bottle[0])

model_cls_label = load_model("model_set/model_cls_label_ResNet18.h5")
classes_cls_label = { 0:'GOOD',
                      1:'ERROR',}

def load_image_cls_label(img):
    img = cv2.resize(img,(90,90))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255   
    # imshow expects values in the range [0, 1]
    return img_tensor

def predict_cnn_label(img):
    img = load_image_cls_label(img)
    pred_label = model_cls_label.predict(img,verbose = False)
    value = classes_cls_label[np.argmax(pred_label)]
    return value, max(pred_label[0])