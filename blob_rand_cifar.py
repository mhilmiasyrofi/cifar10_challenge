"""
Implementation of attack methods. Running this file as a program will
apply the attack to the model specified by the config file and store
the examples in an .npy file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from scipy.spatial.distance import pdist, squareform
from matplotlib import pyplot as plt
import os

import cifar10_input


c = 1.1
import time
class LinfBLOBAttack:
  def __init__(self, model, epsilon, k, a, random_start, loss_func):
    """Attack parameter initialization. The attack performs k steps of
       size a, while always staying within epsilon from the initial
       point."""
    self.model = model
    self.epsilon = epsilon
    self.k = k
    self.a = a
    self.rand = random_start
    self.loss_func = loss_func

    if loss_func == 'xent':
      loss = model.y_xent
      c = 1.1
      print('c: ', c)
    elif loss_func == 'cw':
      label_mask = tf.one_hot(model.y_input,
                              10,
                              on_value=1.0,
                              off_value=0.0,
                              dtype=tf.float32)
      correct_logit = tf.reduce_sum(label_mask * model.pre_softmax, axis=1)
      wrong_logits = (1-label_mask) * model.pre_softmax - label_mask*1e4
      wrong_logit = tf.reduce_max(wrong_logits, axis=1)

      loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
      c = 10.0
      print('c: ', c)
    else:
      print('Unknown loss function. Defaulting to cross-entropy')
      loss = model.y_xent
      c = 1.1
      print('c: ', c)
      
    self.grad = tf.gradients(loss, model.x_input)[0]

  def perturb(self, x_nat, x_adv, y, sess):
    """Given a set of examples (x_nat, y), returns a set of adversarial
       examples within epsilon of x_nat in l_infinity norm."""
      
    batch_size = x_adv.shape[0]  
    
    for epoch in range(10):

#       x_adv = np.array(x_adv).reshape(x_adv.shape[0], 32,32,3)
    
      grad = sess.run(self.grad, feed_dict={self.model.x_input: np.array(x_adv).reshape(x_adv.shape[0], 32,32,3),
                                           self.model.y_input: y})
                                         
      grad = np.array(grad).reshape(grad.shape[0], 32 * 32 * 3)
      kxy, dxkxy = self.svgd_kernel(x_adv)
#       kxy, dxkxy = self.svgd_kernel(np.array(x_adv).reshape(x_adv.shape[0], 32,32,3))
#       print("kxy shape", kxy.shape)
#       print("grad shape", grad.shape)
      x_adv += self.a * np.sign(c*(-(np.matmul(kxy, -grad) + dxkxy)/batch_size) + grad)
  
        
      x_adv = np.clip(x_adv, x_nat - self.epsilon, x_nat + self.epsilon) 
      x_adv = np.clip(x_adv, 0, 255) # ensure valid pixel range


    return x_adv

    
  def svgd_kernel(self, theta):
#     print("Theta: ", theta.shape)
    sq_dist = pdist(theta)
    pairwise_dists = squareform(sq_dist)**2
    
    h = np.median(pairwise_dists)  
    h = np.sqrt(0.5 * h / np.log(theta.shape[0]))

    # compute the rbf kernel
    Kxy = np.exp( -pairwise_dists / h**2 / 2)

    dxkxy = -np.matmul(Kxy, theta)
    sumkxy = np.sum(Kxy, axis=1)
    for i in range(theta.shape[1]):
      dxkxy[:, i] = dxkxy[:,i] + np.multiply(theta[:,i],sumkxy)
    dxkxy = dxkxy / (h**2)
    return (Kxy, dxkxy)
    

if __name__ == '__main__':
  import json
  import sys
  import math


  from model import Model

  with open('config.json') as config_file:
    config = json.load(config_file)

  model_file = tf.train.latest_checkpoint(config['model_dir'])
  if model_file is None:
    print('No model found')
    sys.exit()

  model = Model(mode='eval')
  attack = LinfBLOBAttack(model,
                         config['epsilon'],
                         config['num_steps'],
                         config['step_size'],
                         config['random_start'],
                         config['loss_func'])
  saver = tf.train.Saver()

  data_path = config['data_path']
  cifar = cifar10_input.CIFAR10Data(data_path)

  with tf.Session() as sess:
    # Restore the checkpoint
    saver.restore(sess, model_file)

    # Iterate over the samples batch-by-batch
    num_eval_examples = config['num_eval_examples']
    eval_batch_size = config['eval_batch_size']
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

    

    print('Iterating over {} batches'.format(num_batches))
    

    x_cifar = np.copy(cifar.eval_data.xs)
    
#     x_temp = np.zeros((x_cifar.shape[0],x_cifar.shape[1] * x_cifar.shape[2]))
    x_temp = np.zeros((x_cifar.shape[0],x_cifar.shape[1] * x_cifar.shape[2] * x_cifar.shape[3]))
    for i in range(len(x_cifar)) :   
        img = x_cifar[i]
#         print(img.shape)
#         img = np.dot(img[...,:3], [0.2989, 0.587, 0.114])
        # x_adv[i] = img
        x_temp[i] = img.flatten()
      
    x_cifar = np.copy(x_temp)
    
    x_adv_final = np.copy(x_cifar)
    print("x_adv_final shape:", x_adv_final.shape)
    
    
    for restart in range(50):
#     for restart in range(1):
      # Initialize permutation
      permutation = np.arange(num_eval_examples)
      idx = np.arange(num_eval_examples)
      # Initialize data
      x_test, y_test = np.copy(x_cifar), np.copy(cifar.eval_data.ys)
      
      
      x_adv = x_test + np.random.uniform(-attack.epsilon, attack.epsilon, x_test.shape)
        
      for epoch in range(int(attack.k/10)):
        np.random.shuffle(idx)
        x_test, x_adv, y_test = x_test[idx], x_adv[idx], y_test[idx]
        permutation = permutation[idx]
      
        for ibatch in range(num_batches):
          bstart = ibatch * eval_batch_size
          bend = min(bstart + eval_batch_size, num_eval_examples)
        
          x_batch = x_test[bstart:bend, :]
          x_batch_adv = x_adv[bstart:bend, :]
          y_batch = y_test[bstart:bend]
      
          x_adv[bstart:bend, :] = attack.perturb(x_batch, x_batch_adv, y_batch, sess)
          
      
      inv_permutation = np.argsort(permutation)
      x_adv = x_adv[inv_permutation]
    
    
    predictions = np.array([None] * x_adv.shape[0])
    
    for ibatch in range(num_batches):
      bstart = ibatch * eval_batch_size
      bend = min(bstart + eval_batch_size, num_eval_examples)

      x_batch_adv = x_adv[bstart:bend, :]
      y_batch = np.copy(cifar.eval_data.ys)[bstart:bend]

      predictions[bstart:bend] = sess.run(attack.model.correct_prediction, 
                                             feed_dict={
                                                 attack.model.x_input: np.array(x_batch_adv).reshape(x_batch_adv.shape[0], 32,32,3), 
                                                 attack.model.y_input: y_batch
                                             })
    ## Replace with wrong sample
    print("Final Prediction")
    for i in range(predictions.shape[0]):
      if not predictions[i]:
        x_adv_final[i] = x_adv[i]  

    print('Storing examples')
    path = config['store_adv_path']
    
    np.save(path, np.array(x_adv_final).reshape(x_adv_final.shape[0], 32,32,3))
    print('Examples stored in {}'.format(path))
