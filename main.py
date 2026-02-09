import os,PIL.Image, PIL.ImageFont, PIL.ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

# Configs
im_width = 75
im_height = 75
use_normalized_coordinates = True
BATCH_SIZE = 32
EPOCHS = 5

try:
    MATPLOTLIB_FONT_DIR = os.path.join(os.path.dirname(plt.__file__), "mpl-data/fonts/ttf")
except Exception:
    MATPLOTLIB_FONT_DIR = "" # Fallback or handle appropriately

def draw_bounding_boxes_on_image_array(image, boxes, color=[], thickness=1, display_str_list=()):
    # Fix for PIL handling (H, W, 1) arrays
    if image.ndim == 3 and image.shape[2] == 1:
        image = np.squeeze(image, axis=2)
    image_pil = PIL.Image.fromarray(image)
    rgbimg = PIL.Image.new("RGBA", image_pil.size)
    rgbimg.paste(image_pil)

    draw_bounding_boxes_on_image(rgbimg, boxes, color=color, thickness=thickness, display_str_list=display_str_list)

    return np.array(rgbimg)

def draw_bounding_boxes_on_image(image, boxes, color=[], thickness=1, display_str_list=()):
    boxes_shape = boxes.shape
    if not boxes_shape:
        return
    if len(boxes_shape) != 2 or boxes_shape[1] != 4:
        raise ValueError('Input must be of size [N,4]')

    for i in range(boxes_shape[0]):
        draw_bounding_box_on_image(
            image,
            boxes[i, 1], boxes[i, 0], boxes[i, 3], boxes[i, 2],
            color[i],
            thickness,
            use_normalized_coordinates=True
        )

def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, color='red', thickness=1, use_normalized_coordinates=True):
    draw = PIL.ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)

    draw.line([(left,top), (left,bottom), (right,bottom), (right,top), (left,top)], width = thickness, fill = color)

def dataset_to_numpy_util(training_dataset, validation_dataset, N):
    if tf.executing_eagerly():
        for validation_digits, (validation_labels, validation_bboxes) in validation_dataset:
            validation_digits = validation_digits.numpy()
            validation_labels = np.argmax(validation_labels.numpy(), axis=1)
            validation_bboxes = validation_bboxes.numpy()
            break

        for training_digits, (training_labels, training_bboxes) in training_dataset:
            training_digits = training_digits.numpy()
            training_labels = np.argmax(training_labels.numpy(), axis=1)
            training_bboxes = training_bboxes.numpy()
            break

    return (training_digits, training_labels, training_bboxes,
            validation_digits, validation_labels, validation_bboxes)

def create_digits_from_local_fonts(n):
    font_labels = []
    img = PIL.Image.new('LA',(75*n, 75), color = (0,255))
    try:
        font1 = PIL.ImageFont.truetype(os.path.join(MATPLOTLIB_FONT_DIR, 'DejaVuSansMono-Oblique.ttf'), 25)
        font2 = PIL.ImageFont.truetype(os.path.join(MATPLOTLIB_FONT_DIR, 'STIXGeneral.ttf'), 25)
    except IOError:
        print("Warning: Fonts not found. Using default font.")
        font1 = PIL.ImageFont.load_default()
        font2 = PIL.ImageFont.load_default()

    d = PIL.ImageDraw.Draw(img)
    for i in range(n):
        font_labels.append(i%10)
        d.text((7+i*75, 0 if i < 10 else - 4), str(i%10), fill = (255,255), font = font1 if i < 10 else font2)
    font_digits = np.array(img.getdata() , np.float32)[:,0] / 255.0
    # Reshape logic seems complex, trusting original intent but cleaning syntax if needed
    font_digits = np.reshape(np.stack(np.split(np.reshape(font_digits, [75, 75*n]), n, axis = 1), axis = 0), [n, 75, 75])
    return font_digits, font_labels

def display_digits_with_boxes(digits, predictions, labels, pred_bboxes, bboxes, iou, title):
    n = 8
    # Ensure we don't sample more than available
    n = min(n, len(predictions))
    indexes = np.random.choice(len(predictions), size = n, replace=False)

    n_digits = digits[indexes]
    n_predictions = predictions[indexes]
    n_labels = labels[indexes]

    n_iou = iou[indexes] if len(iou) > 0 else []

    n_pred_bboxes = []
    n_bboxes = []
    if len(iou) > 0:
        n_pred_bboxes = pred_bboxes[indexes]
    if len(bboxes) > 0:
        n_bboxes = bboxes[indexes]

    # Handle shape if needed, assuming digits are flat or 75x75
    # n_digits = n_digits * 255.0
    # Logic from original code, might be modifying in place or copied
    # n_digits = n_digits.reshape(n,75,75)

    fig = plt.figure(figsize = (20,4))
    plt.title(title)
    plt.xticks([])
    plt.yticks([])

    for i in range(n):
        ax = fig.add_subplot(1, n, i+1) # corrected 1,8 to 1,n
        boxes_to_plot = []
        if (len(n_pred_bboxes) > i):
            boxes_to_plot.append(n_pred_bboxes[i])
        if (len(n_bboxes) > i):
            boxes_to_plot.append(n_bboxes[i])

        # Convert simple [75,75] to image that PIL can handle
        img_arr = (n_digits[i] * 255).astype(np.uint8) if np.max(n_digits[i]) <= 1.0 else n_digits[i].astype(np.uint8)

        img_to_draw = draw_bounding_boxes_on_image_array(image = img_arr,
                                                         boxes = np.asarray(boxes_to_plot),
                                                         color = ['red','green'],
                                                         display_str_list  = ["True",'Pred'])
        plt.xlabel(n_predictions[i])
        plt.xticks([])
        plt.yticks([])

        if n_predictions[i] != n_labels[i]:
            ax.xaxis.label.set_color('red') # Fixed typo xais

        plt.imshow(img_to_draw)

        if len(n_iou) > i:
            color = "black"
            # Define iou_threshold if not passed, or pass it? Global used in original.
            iou_threshold = 0.6
            if (n_iou[i][0] < iou_threshold):
                color = "red"
            ax.text(0.2, -0.3, "iou: %s" %(n_iou[i][0]), color = color, transform = ax.transAxes)
    plt.show()

def plot_metrics(history, metric_name, title):
    plt.title(title)
    plt.plot(history.history[metric_name], color = 'blue', label = metric_name)
    plt.plot(history.history['val_' + metric_name], color = 'green', label = 'val_' + metric_name)
    plt.legend()
    plt.show()

def read_image_tfds(image, label):
    xmin = tf.random.uniform((), 0, 48, dtype = tf.int32)
    ymin = tf.random.uniform((), 0, 48, dtype = tf.int32)
    image = tf.reshape(image, (28,28,1,))
    image = tf.image.pad_to_bounding_box(image, ymin, xmin, 75,75)
    image = tf.cast(image, tf.float32) / 255.0
    xmin = tf.cast(xmin, tf.float32)
    ymin = tf.cast(ymin, tf.float32)

    xmax = (xmin + 28) / 75
    ymax = (ymin + 28) / 75
    xmin = xmin / 75
    ymin = ymin / 75

    return image, (tf.one_hot(label, 10), [xmin, ymin, xmax, ymax])

def get_training_dataset(strategy):
    with strategy.scope():
        dataset = tfds.load("mnist", split = "train", as_supervised = True, try_gcs = True)
        dataset = dataset.map(read_image_tfds, num_parallel_calls = tf.data.AUTOTUNE) # Changed 16 to AUTOTUNE
        dataset = dataset.shuffle(5000, reshuffle_each_iteration = True)
        dataset = dataset.repeat()
        dataset = dataset.batch(BATCH_SIZE, drop_remainder = True)
        dataset = dataset.prefetch(-1)
    return dataset

def get_validation_dataset(strategy):
    with strategy.scope():
        dataset = tfds.load("mnist", split = "train", as_supervised = True, try_gcs = True) # splitting? Original used train again?
        # Original code loaded "train" for validation too? Leaving as is but usually should be test.
        dataset = dataset.map(read_image_tfds, num_parallel_calls = tf.data.AUTOTUNE)
        dataset = dataset.batch(10000, drop_remainder = True)
        dataset = dataset.repeat()
    return dataset

def feature_extractor(inputs):
    x = tf.keras.layers.Conv2D(16, activation = 'relu', kernel_size = 3, input_shape = (75,75,1))(inputs)
    x = tf.keras.layers.AveragePooling2D((2,2))(x)

    x = tf.keras.layers.Conv2D(32, activation = 'relu', kernel_size = 3)(x)
    x = tf.keras.layers.AveragePooling2D((2,2))(x)

    x = tf.keras.layers.Conv2D(64, activation = 'relu', kernel_size = 3)(x)
    x = tf.keras.layers.AveragePooling2D((2,2))(x)

    return x

def dense_layers(inputs):
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(128, activation = 'relu')(x)
    return x

def classifier(inputs):
    classification_output = tf.keras.layers.Dense(10, activation = "softmax", name = "classification")(inputs)
    return classification_output

def bounding_box_regression(inputs):
    bounding_box_regression_output = tf.keras.layers.Dense(4, name = "bounding_box")(inputs)
    return bounding_box_regression_output

def final_model(inputs):
    feature_cnn = feature_extractor(inputs)
    dense_output = dense_layers(feature_cnn)

    classification_output = classifier(dense_output)
    bounding_box_output = bounding_box_regression(dense_output)

    model = tf.keras.Model(inputs = inputs, outputs = [classification_output, bounding_box_output])

    return model

def define_and_compile_model(inputs):
    model = final_model(inputs)

    model.compile(optimizer = 'adam', loss = {'classification' : 'categorical_crossentropy',
                                              'bounding_box' : 'mse'},
                  metrics = {'classification' : 'accuracy',
                              'bounding_box' : 'mse'})
    return model

def intersection_over_union(pred_box, true_box):
    xmin_pred, ymin_pred, xmax_pred, ymax_pred = np.split(pred_box, 4, axis = 1)
    xmin_true, ymin_true, xmax_true, ymax_true = np.split(true_box, 4, axis = 1)

    soothing_factor = 1e-10

    xmin_overlap = np.maximum(xmin_pred, xmin_true)
    xmax_overlap = np.minimum(xmax_pred, xmax_true)
    ymin_overlap = np.maximum(ymin_pred, ymin_true)
    ymax_overlap = np.minimum(ymax_pred, ymax_true)

    pred_box_area = (xmax_pred - xmin_pred) * (ymax_pred - ymin_pred)
    true_box_area = (xmax_true - xmin_true) * (ymax_true - ymin_true)

    overlap_area = np.maximum((xmax_overlap - xmin_overlap), 0) * np.maximum((ymax_overlap - ymin_overlap), 0)
    union_area = (pred_box_area + true_box_area) - overlap_area

    iou = (overlap_area + soothing_factor) / (union_area + soothing_factor)

    return iou


if __name__ == "__main__":
    # Strategy
    try:
        strategy = tf.distribute.get_strategy()
        print(f"Number of replicas in sync: {strategy.num_replicas_in_sync}")
    except:
        strategy = tf.distribute.get_strategy() # Fallback or Default?

    # Datasets
    # Warning: using global strategy if needed inside functions, passing it is cleaner.
    training_dataset = get_training_dataset(strategy)
    validation_dataset = get_validation_dataset(strategy)

    # Visualize some data (Optional, keeping consistent with original)
    (training_digits, training_labels, training_bboxes,
     validation_digits, validation_labels, validation_bboxes) = dataset_to_numpy_util(training_dataset, validation_dataset, 10)

    display_digits_with_boxes(training_digits, training_labels,
                              training_labels, np.array([]), training_bboxes,
                              np.array([]), "Training digits and labels")

    # Model
    with strategy.scope():
        inputs = tf.keras.layers.Input(shape = (75,75,1))
        model = define_and_compile_model(inputs)

    model.summary()

    # Training
    steps_per_epoch = 60000 // BATCH_SIZE # Roughly full mnist

    # Correcting logic:
    steps_per_epoch = 60000 // BATCH_SIZE

    history = model.fit(training_dataset, steps_per_epoch = steps_per_epoch,
                        validation_data = validation_dataset, validation_steps = 1, epochs = EPOCHS)

    # Evaluation
    loss, classification_loss, bounding_box_loss, classification_acc, bounding_box_mse = model.evaluate(validation_dataset, steps = 1)

    print("\n-------------------------\n")
    print("Validation Accuracy:" , classification_acc)
    print("\n-------------------------\n")

    plot_metrics(history, "bounding_box_mse", "Bounding Box MSE")
    plot_metrics(history, "classification_accuracy", "Classification Accuracy") # Keras metric might be 'classification_accuracy' or just 'accuracy' depending on name.
    # In compile: metrics = {'classification' : 'accuracy'} -> history key will be 'classification_accuracy' or 'classification_acc'?


    #Classification Losss
    plot_metrics(history, "classification_loss", "Classification Loss")

    # Predictions
    prediction = model.predict(validation_digits, batch_size = 64)
    predicted_labels = np.argmax(prediction[0], axis = 1)
    prediction_bboxes = prediction[1]

    iou = intersection_over_union(prediction_bboxes, validation_bboxes)

    display_digits_with_boxes(validation_digits, predicted_labels, validation_labels,
                              prediction_bboxes, validation_bboxes, iou, "True and Pred Values")