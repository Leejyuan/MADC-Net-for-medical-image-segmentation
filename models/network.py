# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 15:30:27 2021

@author: ljy
"""
from __future__ import division
import os,time,cv2
from tflearn.layers.conv import global_avg_pool
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tf_slim as slim
import numpy as np

def conv_block(inputs, n_filters, name_conv,name_relu,kernel_size=[3, 3], dropout_p=0.0):
	"""
	Basic conv block for Encoder-Decoder
	Apply successivly Convolution, BatchNormalization, ReLU nonlinearity
	Dropout (if dropout_p > 0) on the inputs
	"""
	conv = slim.conv2d(inputs, n_filters, kernel_size, padding='SAME',activation_fn=None, normalizer_fn=None,scope=name_conv)
	out = tf.nn.relu(slim.batch_norm(conv, fused=True),name=name_relu)
	if dropout_p != 0.0:
	  out = slim.dropout(out, keep_prob=(1.0-dropout_p))
	return out

def conv_en_res(inputs,n_filters,stride,name_conv,kernel_size=[3, 3]):

    conv = slim.conv2d(inputs, n_filters, kernel_size, stride, padding='SAME',activation_fn=None, normalizer_fn=None,scope=name_conv)
    out = tf.nn.relu(slim.batch_norm(conv, fused=True))
    return out
def Global_Average_Pooling(x):
    return global_avg_pool(x, name='Global_avg_pooling')

def Relu(x):
    return tf.nn.relu(x)

def Sigmoid(x) :
    return tf.nn.sigmoid(x)

def Fully_connected(x, units, layer_name) :
    with tf.name_scope(layer_name) :
        return tf.layers.dense(inputs=x, use_bias=False, units=units)

def Squeeze_excitation_layer(input_x, out_dim, ratio, layer_name):
    with tf.name_scope(layer_name) :
        squeeze = Global_Average_Pooling(input_x,)
        excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name+'_fully_connected1')
        excitation = Relu(excitation)
        excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name+'_fully_connected2')
        excitation = Sigmoid(excitation)
        excitation = tf.reshape(excitation, [-1,1,1,out_dim])
        scale = input_x * excitation
        return scale
    
def Squeeze_excitation_layer2(input_x, out_dim, ratio, layer_name):
    with tf.name_scope(layer_name) :
        squeeze = Global_Average_Pooling(input_x)
        excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name+'_fully_connected1')
        excitation = Relu(excitation)
        excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name+'_fully_connected2')
        excitation = Sigmoid(excitation)
        excitation = tf.reshape(excitation, [-1,1,1,out_dim])
#        scale = input_x * excitation
        return excitation

def conv_transpose_block(inputs, n_filters,name_conv,name_relu,kernel_size=[3, 3], dropout_p=0.0):
	"""
	Basic conv transpose block for Encoder-Decoder upsampling
	Apply successivly Transposed Convolution, BatchNormalization, ReLU nonlinearity
	Dropout (if dropout_p > 0) on the inputs
	"""
	conv = slim.conv2d_transpose(inputs, n_filters, kernel_size=[3, 3], stride=[2, 2], activation_fn=None,scope=name_conv)
	out = tf.nn.relu(slim.batch_norm(conv), name=name_relu)
	if dropout_p != 0.0:
	  out = slim.dropout(out, keep_prob=(1.0-dropout_p))
	return out

def attention_block(input_down, input_up, F_int):


	W_g1 = slim.conv2d(input_up, F_int, kernel_size=[1,1], stride=1, padding='SAME')
	W_g2 = slim.batch_norm(W_g1)   

	W_x1 = slim.conv2d(input_down, F_int, kernel_size=[1,1], stride=1, padding='SAME') 
	W_x2 = slim.batch_norm(W_x1)

	psi1 = tf.nn.relu(tf.add(W_g2 , W_x2))        
           
	psi = slim.conv2d(psi1, 1, kernel_size=[1,1], stride=1, padding='SAME')
	psi = slim.batch_norm(psi)                       
	psi = tf.sigmoid(psi)        
	out = input_down * psi         
   
	return out


def adptive_weight(DF,channel,rate,layer_name):

    apw=slim.conv2d(DF, 1, [3,3],stride=1, rate=rate, activation_fn=None,padding='SAME')
    acw=Squeeze_excitation_layer2(DF, channel, ratio=4, layer_name=layer_name)
    adptive_weight = tf.matmul(apw,acw)
    adptive_weight = tf.sigmoid(adptive_weight) 
    return adptive_weight
    
    
    
def DF_fusion_block(DF1,DF2,DF3,DF4,size):
	"""
	DF1: 25*25*512
	DF2: 50*50*256
	DF3: 100*100*128
	DF4: 400*400*64
	"""

	DF1_reshape=tf.image.resize_bilinear(DF1, size=size) 
	DF2_reshape=tf.image.resize_bilinear(DF2, size=size)
	DF3_reshape=tf.image.resize_bilinear(DF3, size=size)
	adptive_weight_df1=adptive_weight(DF1_reshape,512,19,layer_name='adptive_weight_df1')
	adptive_df1=DF1_reshape*adptive_weight_df1
	adptive_weight_df2=adptive_weight(DF2_reshape,256,13,layer_name='adptive_weight_df2')
	adptive_df2=DF2_reshape*adptive_weight_df2
	adptive_weight_df3=adptive_weight(DF3_reshape,128,7,layer_name='adptive_weight_df3')
	adptive_df3=DF3_reshape*adptive_weight_df3
	adptive_weight_df4=adptive_weight(DF4,64, 2,layer_name='adptive_weight_df4')
	adptive_df4=DF4*adptive_weight_df4
	DF_fusion=tf.concat(values=[adptive_df1, adptive_df2,adptive_df3,adptive_df4], axis=-1)
	DF_fusion=slim.conv2d(DF_fusion, 64, kernel_size=[1,1], padding='SAME',activation_fn=None, normalizer_fn=None)
	return DF_fusion   
    
    
    
   
def network(inputs, num_classes, preset_model = "NETWORK", dropout_p=0.0, is_training=True, scope=None):
    
   #####################
	# Downsampling path #
	#####################
	label_size=[400,400]
	feature_map_shape1 = [int(x / 2.0) for x in label_size]
	net = conv_block(inputs, 64,name_conv='conv_down_1_1', name_relu='conv_down_relu_1_1') 
	net = conv_block(net, 64,name_conv='conv_down_1_2', name_relu='conv_down_relu_1_2')
	ed_skip_1=net
	en_se_1=Squeeze_excitation_layer(ed_skip_1,64,ratio=4,layer_name='en_se_1')
	net=tf.add(en_se_1, net) 
	ed_reshape1=tf.image.resize_bilinear(net, size=feature_map_shape1) 
	net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX',scope='maxpooling1')

#	res1=net
    
	feature_map_shape2 = [int(x / 4.0) for x in label_size]    
	net = conv_block(net, 128,name_conv='conv_down_2_1', name_relu='conv_down_relu_2_1')
	net = conv_block(net, 128,name_conv='conv_down_2_2', name_relu='conv_down_relu_2_2')
#	net=tf.add(en_se_1, net) 
	net = slim.conv2d(net, 128, kernel_size=[3,3], padding='SAME',activation_fn=None, normalizer_fn=None,scope='scale2')
	en_skip_2 = net
	en_se_2=Squeeze_excitation_layer(en_skip_2,128,ratio=4,layer_name='en_se_2')
	net=tf.add(en_se_2, net) 
	s_attention_1_2=attention_block(ed_reshape1, net, F_int=128 )
	ed_reshape2=tf.image.resize_bilinear(s_attention_1_2, size=feature_map_shape2) 
	ed_skip_2=net
	net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX',scope='maxpooling2')
#	res2=net

#   
	feature_map_shape3 = [int(x / 8.0) for x in label_size] 
	net = conv_block(net, 256,name_conv='conv_down_3_1', name_relu='conv_down_relu_3_1')
	net = conv_block(net, 256,name_conv='conv_down_3_2', name_relu='conv_down_relu_3_2')
	net = conv_block(net, 256,name_conv='conv_down_3_3', name_relu='conv_down_relu_3_3')
	en_skip_3 = net 
	en_se_3=Squeeze_excitation_layer(en_skip_3,256,ratio=4,layer_name='en_c_3')
	net=tf.add(en_se_3, net)

	s_attention_2_3=attention_block(ed_reshape2, net, F_int=256 )
	ed_reshape3=tf.image.resize_bilinear(s_attention_2_3, size=feature_map_shape3) 
	ed_skip_3=net 
	tf.add_to_collection('ed_skip_3', net)
	net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX',scope='maxpooling3')#50
 

	feature_map_shape4 = [int(x / 16.0) for x in label_size] 
	net = conv_block(net, 512 ,name_conv='conv_down_4_1', name_relu='conv_down_relu_4_1')
	net = conv_block(net, 512, name_conv='conv_down_4_2', name_relu='conv_down_relu_4_2')
	net = conv_block(net, 512, name_conv='conv_down_4_3', name_relu='conv_down_relu_4_3')
	en_skip_4 = net 
	en_c_4=Squeeze_excitation_layer(en_skip_4,512,ratio=4,layer_name='en_c_4')
	net=tf.add(en_c_4, net)
 
	s_attention_3_4=attention_block(ed_reshape3, net, F_int=512 )
	ed_reshape4=tf.image.resize_bilinear(s_attention_3_4, size=feature_map_shape4)

	ed_skip_4=net
	net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX',scope='maxpooling4')#25

#    
	net=tf.concat(values=[net,ed_reshape4], axis=-1)    
	net = conv_block(net, 1024,name_conv='conv_bridge_1', name_relu='conv_bridge_relu_1')
	net = conv_block(net, 1024,name_conv='conv_bridge_2', name_relu='conv_bridge_relu_2')#25
    
   #####################
	# Upsampling path #
	#####################

	net = conv_transpose_block(net, 512,name_conv='upsampling4', name_relu='upsampling_relu_4')#50
	attention4=attention_block(ed_skip_4, net, F_int=512)

	net=tf.concat(values=[attention4, net], axis=-1)     
	net = conv_block(net, 512,name_conv='conv_up_4_1', name_relu='conv_up_relu_4_1')
	net = conv_block(net, 512,name_conv='conv_up_4_2', name_relu='conv_up_relu_4_2')
	DF1=net

	net = conv_transpose_block(net, 256,name_conv='upsampling3', name_relu='upsampling_relu_3') #100

	attention3=attention_block(ed_skip_3, net, F_int=256 )
	net=tf.concat(values=[attention3, net], axis=-1) 
	net = conv_block(net, 256,name_conv='conv_up_3_1', name_relu='conv_up_relu_3_1')
	net = conv_block(net, 256,name_conv='conv_up_3_2', name_relu='conv_up_relu_3_2')
	DF2=net  

	net = conv_transpose_block(net, 128,name_conv='upsampling2', name_relu='upsampling_relu_2')
S
	attention2=attention_block(ed_skip_2, net, F_int=128 )
	net=tf.concat(values=[attention2, net], axis=-1)
	net = conv_block(net, 128,name_conv='conv_up_2_1', name_relu='conv_up_relu_2_1')
	net = conv_block(net, 128,name_conv='conv_up_2_2', name_relu='conv_up_relu_2_2')
	DF3=net      
    

	net = conv_transpose_block(net, 64,name_conv='upsampling1', name_relu='upsampling_relu_1')
	attention1=attention_block(ed_skip_1, net, F_int=64)
	net=tf.concat(values=[attention1, net], axis=-1)
	net = conv_block(net, 64,name_conv='conv_up_1_1', name_relu='conv_up_relu_1_1')
	net = conv_block(net, 64,name_conv='conv_up_1_2', name_relu='conv_up_relu_1_2')
	DF4=net 
	DF_fusion=DF_fusion_block(DF1,DF2,DF3,DF4,label_size)
	#####################
	#      Softmax      #
	#####################
	net = slim.conv2d(DF_fusion, num_classes, [1, 1], activation_fn=None, scope='logits')
	tf.add_to_collection('finally', net)	
	return net