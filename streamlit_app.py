import streamlit as st

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import PIL

import pandas as pd
import numpy as np
from typing import Dict

# define layout
st.beta_set_page_config(page_title='PlantPal', page_icon=':seedling:', layout='centered', initial_sidebar_state='expanded')

# initialize uploaded file dictionary
@st.cache(allow_output_mutation=True)
def get_static_store():
    """This list is initialized once and can be used to store the files uploaded"""
    return []

# suppress warning
st.set_option('deprecation.showfileUploaderEncoding', False)

# set NN parameters
img_height = 240
img_width = 240
no_classes = 16

# define classes
class_names = ['Aloe_Vera', 'Asparagus_Fern', 'Baby_Rubber_Plant', 'Boston_Fern',
            'Fiddle_Leaf_Fig', 'Golden_Pothos', 'Jade_Plant', 'Maidenhair_Fern',
            'Monstera', 'Parlor_Palm', 'Peace_Lily', 'Pothos', 'Rubber_Plant',
            'Snake_Plant', 'Spider_Plant', 'Umbrella_Tree']

# Header
logo = PIL.Image.open('./data/streamlit/Logo.png')
st.image(logo, width=700, output_format='PNG') # logo

def predict_image(im):
    resized_im = im
    try:
        np_im = np.array(resized_im)
        np_im = np_im[None, :]
        result = model.predict(np_im)
        index_best = np.argmax(result)
    except ValueError:
        st.error('It appears your image isn\'t of the right format, Please input a valid image')

    return class_names[index_best]

def predict_top5(im):
    resized_im = im
    top5_plants = []
    top5_probs = []
    try:
        np_im = np.array(resized_im)
        np_im = np_im[None, :]
        result = model.predict(np_im)
        top5_indxs = np.argpartition(result, -5)[0][-5:]
        top5_indxs = top5_indxs[::-1]
        for indx in top5_indxs:
            top5_plants.append(class_names[indx])
            top5_probs.append(result[0][indx])
    except ValueError:
        st.error('It appears your image isn\'t of the right format, Please input a valid image')
    return top5_plants, top5_probs

def get_top5_confirmation(top5_plants, top5_probs):

    top5_plants_spaced = [s.replace('_', ' ') for s in top5_plants] # replace _ with a space
    top5_df = plants_df[plants_df['name'].isin(top5_plants_spaced)] # isolate relevant plant data

    # append softmax values to dataframe
    top5_df['softmax'] = 0
    for i in np.arange(5):
        top5_df.loc[top5_df['name'] == top5_plants_spaced[i], ['softmax']] = top5_probs[i]

    top5_df.sort_values(by='softmax', ascending=False, inplace=True)
    top5_df.reset_index(inplace=True, drop=True)

    confirmed_indx = display_5_pick_1(top5_df)
    confirmed_plant = top5_df['name'].iloc[confirmed_indx]

    print_info(confirmed_plant, im)

def find_key_true(input_dict):
    return next((k for k, v in input_dict.items() if v == True), None)

def display_5_pick_1(top5_df):

    button_dict = {0: False,
                   1: False,
                   2: False,
                   3: False,
                   4: False}

    while all(value == False for value in button_dict.values()) == True:

        st.write('## These are the 5 most likely plants, select the one ')

        columns = st.beta_columns(5)

        for i, col in enumerate(columns):

            # get softmax
            plant_softmax = top5_df.iloc[i]['softmax']
            softmax_neat = '{:4.4} %'.format(plant_softmax * 100)

            # display token image
            token_img = PIL.Image.open(f'./data/streamlit/class_imgs/{top5_df.iloc[i].img_file}')
            token_img = token_img.resize((400,500)) # make image sizes uniform
            col.image(token_img, width=140, output_format='PNG', caption=softmax_neat)  # logo

            # get name
            plant_name = top5_df.iloc[i]['name']
            button_dict[i] = col.button(f'{plant_name}')

    return find_key_true(button_dict)

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
    plants_df = pd.read_csv('./data/streamlit/house_plants_16c.csv')
    return plants_df

def print_info(plant, im):
    # print image
    st.image(im, width=700)

    # print info
    plant_df = plants_df[plants_df['name'] == plant]
    st.markdown(f"<h1 style='text-align: center; color: black;'>{plant}</h1>", unsafe_allow_html=True)

    # latin name hyperlink
    latin_name = plant_df['latin_name'].values[0]
    wiki_url = plant_df['link'].values[0]

    # centered
    st.markdown(
        f"<a style='display: block; text-align: center;' href={wiki_url}>{latin_name}</a>",
        unsafe_allow_html=True,
    )

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

# img = PIL.Image.open('./data/streamlit/download.jpeg')
# st.sidebar.image(img)

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
        # if len(uploaded_files) == 1:
        #
        top5_plants, top5_probs = predict_top5(uploaded_files_resized[0])
        get_top5_confirmation(top5_plants, top5_probs)

        # name = predict_image(uploaded_files_resized[0])
        # print_info(name.replace('_', ' '), uploaded_files[0])
