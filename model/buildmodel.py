import tensorflow as tf

def build_model(input=(512, 512, 3), nbrclass=2, backbone='resnet101', lr=0.0001):
    import segmentation_models as sm
    sm.set_framework('tf.keras')
    sm.framework()
    model = sm.Unet(backbone,
                classes=nbrclass, 
                input_shape = input, 
                activation='softmax', 
                encoder_freeze=False)
    model.compile(optimizer = tf.keras.optimizers.Adam(lr),
              loss = sm.losses.DiceLoss(),
              metrics = [sm.metrics.iou_score, sm.metrics.f1_score])
    return model

def load_model(model, path):
    model.load_weights(path)
    return model
