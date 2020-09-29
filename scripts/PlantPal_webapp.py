import streamlit as st

import tensorflow as tf

import PIL
import cv2

import pandas as pd

# suppress warning
st.set_option('deprecation.showfileUploaderEncoding', False)


def print_info(plant):
    print('inside print info...')
    print(plant)
    print(plants_df)
    # plant_df=plants_df[['Common Name']==plant].copy()
    # plant_df = plants_df.query('Common Name == plant').copy()
    plant_df = plants_df[plants_df['Common Name'] == plant]

    name_str = '{}'.format(plant_df['Common Name'].values[0])
    st.markdown(f"<h1 style='text-align: center; color: black;'>{name_str}</h1>", unsafe_allow_html=True)
    # st.write('\n**NAME:** *{}*'.format(plant_df['Common Name'].values[0]))

    st.markdown('---')

    st.write('**CARE:**')
    st.write('**Water**: *{}*'.format(plant_df['Water'].values[0]))
    st.write('**Light**: *{}*'.format(plant_df['Light'].values[0]))

    st.markdown('---')

    st.write('**PET SAFETY:**')
    st.write('**Cats**: *{}*'.format(plant_df['Cat Safe'].values[0]))
    st.write('**Dogs**: *{}*'.format(plant_df['Dog Safe'].values[0]))

    st.markdown('---')


def cv2_imread(path, label):
    # read in the image, getting the string of the path via eager execution
    img = cv2.imread(path.numpy().decode('utf-8'), cv2.IMREAD_UNCHANGED)
    # change from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, label


def tf_cleanup(img, label):
    # convert to Tensor
    img = tf.convert_to_tensor(img)
    # unclear why, but the jpeg is read in as uint16 - convert to uint8
    img = tf.dtypes.cast(img, tf.uint8)
    # set the shape of the Tensor
    img.set_shape((128, 128, 3))
    # convert to float32, scaling from uint8 (0-255) to float32 (0-1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image
    img = tf.image.resize(img, [128, 128])
    # convert the labels into a Tensor and set the shape
    label = tf.convert_to_tensor(label)
    label.set_shape(())
    return img, label


def create_model(img_shape=(128, 128, 3)):
    img_shape = img_shape

    # create the base model from the pre-trained model VGG16
    # note that, if using a Kaggle server, internet has to be turned on
    pretrained_model = tf.keras.applications.vgg16.VGG16(input_shape=img_shape,
                                                         include_top=False,
                                                         weights='imagenet')

    # freeze the convolutional base
    pretrained_model.trainable = False
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(1,
                                             kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1992),
                                             bias_initializer=tf.keras.initializers.GlorotUniform(seed=1992))

    model = tf.keras.Sequential([pretrained_model,
                                 global_average_layer,
                                 prediction_layer])

    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # unfreeze the layers
    pretrained_model.trainable = True

    # fine-tune from this layer onwards
    fine_tune_at = 15

    # freeze all the layers before the `fine_tune_at` layer
    for layer in pretrained_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate / 10),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def predict_plant(file):
    input_ds = tf.data.Dataset.from_tensor_slices(([file], [1]))

    # map the cv2 wrapper function using `tf.py_function`
    input_ds = input_ds.map(
        lambda path, label: tuple(tf.py_function(cv2_imread, [path, label], [tf.uint16, label.dtype])),
        num_parallel_calls=autotune)

    # map the TensorFlow transformation function - no need to wrap
    input_ds = input_ds.map(tf_cleanup, num_parallel_calls=autotune)

    # check that the image was read in correctly
    for image, label in input_ds.take(1):
        print("image shape : ", image.numpy().shape)
        print("label       : ", label.numpy())

    input_batches_ds = input_ds.batch(1)

    result = model.predict(input_batches_ds)

    return result[0][0]


# preload model and information
plants_df = pd.read_csv('../data/House_Plants.csv')
model = create_model()
model.load_weights('../models/binary_prelimmod_1.0_weights.h5')
autotune = tf.data.experimental.AUTOTUNE

st.write("""
         # PlantPal!
         """
         )

uploaded_file = st.file_uploader("Please upload an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = PIL.Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    saved_path: str = '../uploaded_imgs/file1.jpeg'
    image.save(saved_path)

    prediction = predict_plant(saved_path)

    if prediction < 0:
        name = 'Fiddle Leaf Fig'
    else:
        name = 'Aloe Vera'

    # st.write(f"Your plant is a {name}")
    print_info(name)

st.write("\n\n\nP.S. Insight, you guys rock!")
