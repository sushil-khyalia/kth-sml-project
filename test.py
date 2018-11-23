from __future__ import print_function
import torch
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
import utils
from models import CNNEncoder, DecoderRNN
from vocab import Vocabulary, load_vocab
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def main(args):
  
  vocab = load_vocab()
  
  encoder = CNNEncoder()
  decoder = DecoderRNN(512,512,len(vocab))
  
  encoder_state_dict, decoder_state_dict, optimizer, *meta = utils.load_models(args.checkpoint_file,False)
  encoder.load_state_dict(encoder_state_dict)
  decoder.load_state_dict(decoder_state_dict)
  
  if torch.cuda.is_available():
    encoder.cuda()
    decoder.cuda()
    
  transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
  
  inp = cv2.imread(args.image_path)
  inp = transform(Image.fromarray(inp)).unsqueeze(0)
  inp = utils.to_var(inp, volatile=True)
  
  features = encoder(inp)
  sampled_ids = decoder.sample(features)
  
  sampled_ids = sampled_ids.cpu().data.numpy()[0]
  sentence = utils.convert_back_to_text(sampled_ids, vocab)
  
  print('Caption:', sentence)
  
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_file', type=str,
            default=None, help='path to saved checkpoint')
    parser.add_argument('--image_path', type=str,
            default=None, help='path to the input image')
    args = parser.parse_args()
    main(args)