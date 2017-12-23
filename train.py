from __future__ import print_function
import torch
from torchvision import datasets, models, transforms
from torchvision.utils import make_grid
from torchvision import transforms
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
import torch.optim as optim
import torch.nn as nn
from torch import np
import utils
from data_loader import get_coco_data_loader
from models import CNN, RNN
from vocab import Vocabulary, load_vocab
import os

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def main():
    # hyperparameters
    batch_size = 32
    num_workers = 1
    cnn_output_dim = 1001

    # Image Preprocessing
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), 
                                     (0.229, 0.224, 0.225))
        ])
    }

    # load COCOs dataset
    IMAGES_PATH = 'data/train2014'
    CAPTION_FILE_PATH = 'data/annotations/captions_train2014.json'

    vocab = load_vocab()
    train_loader = get_coco_data_loader(path=IMAGES_PATH,
                                        json=CAPTION_FILE_PATH,
                                        vocab=vocab,
                                        transform=data_transforms['train'],
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=num_workers)

    # Assumes we extracted Wojtek's devset in data/
    IMAGES_PATH = 'data/data_pack/images/dev2014'
    CAPTION_FILE_PATH = 'data/data_pack/captions_dev2014.json'
    val_loader = get_coco_data_loader(path=IMAGES_PATH,
                                      json=CAPTION_FILE_PATH,
                                      vocab=vocab,
                                      transform=data_transforms['train'],
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=num_workers)


    # show some sample images
    """
    images, captions, lengths = next(iter(train_loader))
    out = make_grid(images[0])
    utils.imshow(out, figsize=(10,6), title=[vocab.idx2word[idx] for idx in captions[0]])

    #input('Press Enter to continue...')
    """

    #images = Variable(images)
    #labels = Variable(captions)

    # load pretrained ResNet18 model
    #original_model = models.resnet18(pretrained=True)
    # TODO: re-write as its own class according to the pytorch tutorials, with proper forward()
    # TODO: this is needed for having variable output_dim
    #model = utils.FeatureExtractor(original_model, output_dim=1001)

    # freeze weights
    #for param in original_model.parameters():
    #    param.requires_grad = False

    #outputs = original_model(images) # batch_size x 1000

    #print(original_model)

    losses_val = []
    losses_train = []

    # Build the models
    embed_size = 256
    num_hiddens = 512
    learning_rate = 0.001
    num_epochs = 3
    log_step = 125
    save_step = 1000
    model_path = 'models'
    encoder = CNN(embed_size)
    decoder = RNN(embed_size, num_hiddens, len(vocab), 1, rec_unit='lstm')
    
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.batchnorm.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    
    # Train the Models
    total_step = len(train_loader)

    for epoch in range(num_epochs):
        for i, (images, captions, lengths) in enumerate(train_loader):
            
            # Set mini-batch dataset
            images = to_var(images, volatile=True)
            captions = to_var(captions)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            
            # Forward, Backward and Optimize
            decoder.zero_grad()
            encoder.zero_grad()
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            train_loss = criterion(outputs, targets)
            losses_train.append(train_loss.data[0])
            train_loss.backward()
            optimizer.step()

            # Print log info
            if i % log_step == 0:
                for step, (images, captions, lengths) in enumerate(val_loader):
                    images = to_var(images, volatile=True)
                    captions = to_var(captions, volatile=True)

                    targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
                    features = encoder(images)
                    outputs = decoder(features, captions, lengths)
                    val_loss = criterion(outputs, targets)
                    losses_val.append(val_loss.data[0])

                print('Epoch: {} - Step: {} - Train Loss: {} - Eval Loss: {}'.format(epoch, step, train_loss.data[0], val_loss.data[0]))
                
            # Save the models
            if (i+1) % save_step == 0:
                torch.save(decoder.state_dict(), 
                           os.path.join(model_path, 
                                        'decoder-%d-%d.pkl' %(epoch+1, i+1)))
                torch.save(encoder.state_dict(), 
                           os.path.join(model_path, 
                                        'encoder-%d-%d.pkl' %(epoch+1, i+1)))

    #"""

if __name__ == '__main__':
    main()