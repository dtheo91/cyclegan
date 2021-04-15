######################################################################
# Setup
from flask import Flask, render_template, flash, request, redirect, url_for, jsonify
import os 
from werkzeug.utils import secure_filename
import transform
import json
import numpy as np 
import cv2
import io
from PIL import Image
import base64
import models.mask_extractor as me
import torch
import torchvision
from torchvision.utils import save_image

app = Flask(__name__)
######################################################################

######################################################################
# Functions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def data_uri_to_cv2_img(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    return img
######################################################################

######################################################################
# App Routes
@app.route('/')
def index():
    return render_template("base.html")
        
@app.route('/somefunction')
def some_func():
    button_id = request.args.get("param_first")
    image = request.args.get("param_second")
    if button_id == "None" or image == "None": 
        return jsonify({"reply": "No Domain or Image"})
    else:
        img_arr = data_uri_to_cv2_img(image)
        
        if button_id == "simpson":
            model = torch.load("generator_simpson.pt")
            model.eval()
        elif button_id == "manga":
            model = torch.load("generator_manga.pt")
            model.eval()
        elif button_id == "cartoon":
            model = torch.load("generator_cartoon.pt")
            model.eval()
        elif button_id == "sketch":
            model = torch.load("generator_sketch.pt")
            model.eval()
        else: 
            raise NotImplemented
        
        mask_extractor = me.BiSeNet(19)
        mask_extractor.load_state_dict(torch.load("79999_iter.pth"))
        mask_extractor.eval()
        
        img_tensor = torchvision.transforms.ToTensor()(img_arr)
        img_tensor = torchvision.transforms.Resize((512,512))(img_tensor)
        print(img_tensor)
        save_image(img_tensor, "imgs/orig_test.jpg")
        img_tensor = img_tensor.unsqueeze(0)
        
        img_tensor_mask = mask_extractor(img_tensor)[0].squeeze(0).detach().argmax(0)

        img_tensor = torchvision.transforms.Resize((256,256))(img_tensor)
        img_tensor_mask = torchvision.transforms.Resize((256,256))(img_tensor_mask.unsqueeze(0))
        
        img_combined = torch.cat([img_tensor_mask, img_tensor.squeeze(0)])
        
        transformed = model.forward(img_combined.unsqueeze(0))
        transformed = (transformed.cpu().detach().squeeze(dim=0)+1)/2
        #print(transformed.shape)
        transformed_pil = torchvision.transforms.ToPILImage()(transformed)
        
        #transformed_pil.save("imgs/test.png")
        img_byte_arr = io.BytesIO()
        #print(transformed_pil.size)
        transformed_pil.save(img_byte_arr, format='PNG')
        my_encoded_img = base64.encodebytes(img_byte_arr.getvalue()).decode('ascii')
    
        return jsonify({"reply": "Success", "image": my_encoded_img})

######################################################################