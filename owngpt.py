# -*- coding: utf-8 -*-
"""
Create ownChat web application streamlit and private gpt
@author: Avinash G
"""
#from dotenv import load_dotenv
import streamlit as st
import os
from fastapi import FastAPI, UploadFile, File
from typing import List, Optional
import urllib.parse

import owngptsettings
import  privategpt


def get_text():
    input_text = st.text_input("Enter Your Text", key="input")
    return input_text 


secret = ''
st.set_page_config(
    page_title="Own ChatGPT App",
    page_icon=":robot:"
)

st.header("Own ChatGPT App Private")

if 'Bot_msg' not in st.session_state:
    st.session_state['Bot_msg'] = []

if 'History_msg' not in st.session_state:
    st.session_state['History_msg'] = []

user_input = get_text()

if user_input:
    st.session_state.History_msg.append(user_input)
    #st.session_state.Bot_msg.append(privategpt.Bot_generate_msg(user_input))
    answer, docs = privategpt.private_gpt_generate_msg(user_input,False)
    st.session_state.Bot_msg.append(answer)

if st.session_state['Bot_msg']:
    for i in range(len(st.session_state['Bot_msg'])-1, -1, -1):
        st.markdown("BOT :- "+" "+st.session_state["Bot_msg"][i])
        st.markdown("HUMAN :- "+"\n"+st.session_state['History_msg'][i])
 
        
