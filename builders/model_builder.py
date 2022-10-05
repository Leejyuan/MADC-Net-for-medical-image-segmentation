import sys, os
import tensorflow as tf
import subprocess

sys.path.append("models")

from models.network import network




def build_model(model_name, net_input, num_classes, crop_width, crop_height, frontend, is_training=True):
	# Get the selected model. 
	# Some of them require pre-trained ResNet

	print("Preparing the model ...")
	MODEL = None

	MODEL = network(net_input,  num_classes=num_classes,preset_model = model_name)	


	return MODEL
