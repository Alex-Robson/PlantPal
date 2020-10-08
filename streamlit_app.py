import streamlit as st

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import PIL
from bokeh.models.widgets import Div

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

# define layout
st.beta_set_page_config(page_title='PlantPal', page_icon=':seedling:', layout='centered', initial_sidebar_state='expanded')

# set parameters
img_height = 240
img_width = 240
no_classes = 16

# define classes
class_names = ['Aloe_Vera', 'Asparagus_Fern', 'Baby_Rubber_Plant', 'Boston_Fern',
            'Fiddle_Leaf_Fig', 'Golden_Pothos', 'Jade_Plant', 'Maidenhair_Fern',
            'Monstera', 'Parlor_Palm', 'Peace_Lily', 'Pothos', 'Rubber_Plant',
            'Snake_Plant', 'Spider_Plant', 'Umbrella_Tree']

# Header
logo = PIL.Image.open('./data/streamlit/ar1.png')
# st.markdown('![logo](./data/streamlit/ar1.png)')
st.image(logo, width=700, output_format='png') # logo


# st.markdown(f"<h1 style='text-align: center; color: black;'>PlantPal</h1>", unsafe_allow_html=True)
# st.markdown(f"<h1 style='text-align: center; color: black;'>For happy plants and healthy pets</h1>", unsafe_allow_html=True)
# st.markdown(f"<font size=30><h1 style='text-align: center; color: black;'>PlantPal</h1></font>", unsafe_allow_html=True)
# st.write("""# PlantPal """ ) # page title

def predict_image(im):
    resized_im = im #.resize((240, 240))
    try:
        np_im = np.array(resized_im)
        np_im = np_im[None, :]
        result = model.predict(np_im)
        index_best = np.argmax(result)
    except ValueError:
        st.error('It appears your image isn\'t of the right format, Please input a valid image')

    st.write(f'{np.sum(result)}')
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
    plants_df = pd.read_csv('./data/house_plants.csv', skiprows=2,
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
    st.image(im, width=700)

    # print info
    plant_df = plants_df[plants_df['name'] == plant]
    st.markdown(f"<h1 style='text-align: center; color: black;'>{plant}</h1>", unsafe_allow_html=True)

    # latin name hyperlink
    latin_name = plant_df['latin_name'].values[0]
    wiki_url = 'www.wikipedia.org'#plant_df['wikilink'].values[0]
    link = f'[{padded_latin_name}]({wiki_url})'
    st.markdown(link, unsafe_allow_html=True)
    # st.markdown("<h1 style='text-align: center; color: red;'>Some title</h1>", unsafe_allow_html=True)

    st.markdown('---')

    st.write('## CARE:')
    st.write('**Water**: *{}*'.format(plant_df['water_lmh'].values[0]))
    st.write('**Light**: *{}*'.format(plant_df['light'].values[0]))

    st.markdown('---')

    st.write('## PET SAFETY:')
    st.write('**Cats**: *{}*'.format(plant_df['cat_tox'].values[0]))
    if plant_df['cat_tox'].values[0] != 'Non-toxic':
        st.write('     {}'.format(plant_df['cat_tox_desc'].values[0]))

    st.write('**Dogs**: *{}*'.format(plant_df['dog_tox'].values[0]))
    if plant_df['dog_tox'].values[0] != 'Non-toxic':
        st.write('     {}'.format(plant_df['dog_tox_desc'].values[0]))

# preload model and information
plants_df = load_plant_df()
model = create_model()
model.load_weights('./models/convmod_1.0_weights.h5')

uploaded_files = [] # initialize list of full resolution uploads
uploaded_files_resized = [] # initialize list of full resolution uploads

####### Sidebar #######
# Pick from list
st.sidebar.write('## Search Method:')
search_option = st.sidebar.selectbox(
    '',
     ['upload an image', 'from a list'])

st.sidebar.write('## Sample Images:')

img = PIL.Image.open('./data/streamlit/download.jpeg')
st.sidebar.image(img)

#######################

if search_option == 'from a list':
    option = st.selectbox(
        'Select a plant by name:',
         plants_df['name'])

    'You selected: ', option

elif search_option == 'upload an image':

    st.write('upload an image of a house plant:')
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Process you file here
        im = PIL.Image.open(uploaded_file)

        if not im in uploaded_files:
            im = PIL.Image.open(uploaded_file) # TIDY THIS
            # im700 = im.resize((700, 700))
            np_im700 = np.array(im)
            uploaded_files.append(np_im700)

            # im700 = im.resize((700, 700))
            # np_im700 = np.array(im700)
            # uploaded_files.append(np_im700)

            im = PIL.Image.open(uploaded_file)
            im240 = im.resize((240, 240))
            np_im240 = np.array(im240)
            uploaded_files_resized.append(np_im240)
    else:
        st.info("Please upload at least one image of a house plant")

    if uploaded_file:
        if len(uploaded_files) == 1:
            name = predict_image(uploaded_files_resized[0])
            print_info(name.replace('_', ' '), uploaded_files[0])
