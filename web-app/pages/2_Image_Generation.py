import streamlit as st
import requests
import json
import matplotlib.pyplot as plt
import numpy as np
import time
import boto3

from configs import *

from PIL import Image

image = Image.open("./img/sagemaker.png")
st.image(image, width=80)
st.header("Image Generation")
st.caption("Using Stable Diffusion model from Hugging Face")

runtime = boto3.client("runtime.sagemaker")


with st.spinner("Retrieving configurations..."):
    all_configs_loaded = False

    while not all_configs_loaded:
        try:
            # api_endpoint = get_parameter(key_txt2img_api_endpoint)
            sm_endpoint = get_parameter(key_txt2img_sm_endpoint)
            all_configs_loaded = True
        except:
            time.sleep(5)

    endpoint_name = st.sidebar.text_input("SageMaker Endpoint Name:", sm_endpoint)
    # url = st.sidebar.text_input("API GW Url:", api_endpoint)


prompt = st.text_area("Input Image description:", """Dog in superhero outfit""")

if st.button("Generate image"):
    if endpoint_name == "" or prompt == "":  # or url == "":
        st.error("Please enter a valid endpoint name and prompt!")
    else:
        with st.spinner("Wait for it..."):
            try:
                response = runtime.invoke_endpoint(
                    EndpointName=endpoint_name,
                    Body=prompt,
                    ContentType="application/x-text",
                )
                response_body = json.loads(response["Body"].read().decode())
                image_array = response_body["generated_image"]
                st.image(np.array(image_array))

            except runtime.exceptions.InternalFailure as erri:
                st.error("InternalFailure:", erri)

            except runtime.exceptions.ServiceUnavailable as errs:
                st.error("ServiceUnavailable:", errs)

            except runtime.exceptions.ValidationError as errv:
                st.error("ValidationError:", errv)

            except runtime.exceptions.ModelError as errm:
                st.error("ModelError", errm)

            except runtime.exceptions.ModelNotReadyException as err:
                st.error("ModelNotReadyException", err)

        st.success("Done!")
