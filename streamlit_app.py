import streamlit as st

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import PIL
import opencv as cv2

import pandas as pd
import numpy as np
from typing import Dict

# initialize uploaded file dictionary
@st.cache(allow_output_mutation=True)
def get_static_store():
    """This list is initialized once and can be used to store the files uploaded"""
    return []

# suppress warning
st.set_option('deprecation.showfileUploaderEncoding', False)
st.beta_set_page_config(page_title=None, page_icon=':seedling:', layout='wide', initial_sidebar_state='auto')

# set parameters
img_height = 240
img_width = 240
no_classes = 16

# define classes
class_names = ['Aloe_Vera', 'Asparagus_Fern', 'Baby_Rubber_Plant', 'Boston_Fern',
            'Fiddle_Leaf_Fig', 'Golden_Pothos', 'Jade_Plant', 'Maidenhair_Fern',
            'Monstera', 'Parlor_Palm', 'Peace_Lily', 'Pothos', 'Rubber_Plant',
            'Snake_Plant', 'Spider_Plant', 'Umbrella_Tree']

# page title
st.write("""
         # PlantPal!
         """
         )

def cv2_imread(path, label):
    # read in the image, getting the string of the path via eager execution
    img = cv2.imread(path.numpy().decode('utf-8'), cv2.IMREAD_UNCHANGED)
    # change from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, label

def predict_image(im):
    resized_im = im #.resize((240, 240))
    try:
        np_im = np.array(resized_im)
        np_im = np_im[None, :]
        result = model.predict(np_im)
        index_best = np.argmax(result)
    except ValueError:
        st.error('It appears your image isn\'t of the right format, Please input a valid image')
    return class_names[index_best]

def create_model():

    model = Sequential([
        layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(64, 7, activation='relu', padding='same'),
        layers.MaxPooling2D(2),
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(2),
        layers.Conv2D(256, 3, activation='relu', padding='same'),
        layers.Conv2D(256, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(no_classes, activation='softmax')
    ])
    return model

def load_plant_df():
    plants_df = pd.read_csv('../data/House_Plants.csv', skiprows=2,
                            names=['name', 'type', 'latin_name',
                                   'name_1', 'name_2', 'name_3',
                                   'water_lmh', 'water_desc',
                                   'light', 'issues', 'fert',
                                   'cat_tox', 'cat_tox_desc',
                                   'dog_tox', 'dog_tox_desc',
                                   'fun_fact'])
    return plants_df

def print_info(plant, im):
    # print image
    st.image(im)

    # print info
    plant_df = plants_df[plants_df['name'] == plant]
    st.markdown(f"<h1 style='text-align: center; color: black;'>{plant}</h1>", unsafe_allow_html=True)

    st.markdown('---')

    st.write('**CARE:**')
    st.write('**Water**: *{}*'.format(plant_df['water_lmh'].values[0]))
    st.write('**Light**: *{}*'.format(plant_df['light'].values[0]))

    st.markdown('---')

    st.write('**PET SAFETY:**')
    st.write('**Cats**: *{}*'.format(plant_df['cat_tox'].values[0]))
    if plant_df['cat_tox'].values[0] != 'Non-toxic':
        st.write('     {}'.format(plant_df['cat_tox_desc'].values[0]))

    st.write('**Dogs**: *{}*'.format(plant_df['dog_tox'].values[0]))
    if plant_df['dog_tox'].values[0] != 'Non-toxic':
        st.write('     {}'.format(plant_df['dog_tox_desc'].values[0]))

    # st.markdown('---')

def plant_col(plant, im, col):
    # print image
    col.image(im)

    # print plant info
    plant_df = plants_df[plants_df['name'] == plant]
    col.markdown(f"<h1 style='text-align: center; color: black;'>{plant}</h1>", unsafe_allow_html=True)

    col.markdown('---')

    col.write('**CARE:**')
    col.write('**Water**: *{}*'.format(plant_df['water_lmh'].values[0]))
    col.write('**Light**: *{}*'.format(plant_df['light'].values[0]))

    col.markdown('---')

    col.write('**PET SAFETY:**')
    if plant_df['cat_tox'].values[0] != 'Non-toxic':
        col.write('**Cats**: *{}*'.format(plant_df['cat_tox'].values[0]))
        col.write('     {}'.format(plant_df['cat_tox_desc'].values[0]))

    col.write('**Dogs**: *{}*'.format(plant_df['dog_tox'].values[0]))
    if plant_df['dog_tox'].values[0] != 'Non-toxic':
        col.write('     {}'.format(plant_df['dog_tox_desc'].values[0]))

    # col.markdown('---')

# preload model and information
plants_df = load_plant_df()
model = create_model()
model.load_weights('../models/convmod_1.0_weights.h5')

uploaded_files = []

static_store = get_static_store()
uploaded_file = st.file_uploader("Please upload an image of one of your house plants", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Process you file here
    im = PIL.Image.open(uploaded_file)

    # And add it to the static_store if not already in
    if not im in static_store:
        im = PIL.Image.open(uploaded_file)
        resized_im = im.resize((240, 240))
        np_im = np.array(resized_im)
        static_store.append(np_im)
else:
    static_store.clear()  # Hack to clear list if the user clears the cache and reloads the page
    st.info("Upload another image")

# if st.button("Clear file list"):
#     static_store.clear()

if uploaded_file:
    if len(static_store) == 1:
        name = predict_image(static_store[0])
        print_info(name.replace('_', ' '), static_store[0])
    else:
        columns = st.beta_columns(len(static_store))
        for i, col in enumerate(columns):
            name = predict_image(static_store[i])
            plant_col(name.replace('_', ' '), static_store[i], col)
