

batch_size: 128
vocab_size: 15000
CNN resnet34, pretrained on imagenet
lr: 2e-4
embed_size
    group: 256,512
log_step: 125

Experiments:  
Comparing Basic_RNN, LSTM and GRU    


# Neural Image Captioning
The goal of this project was to tackle the problem of automatic caption generation for images. As part of the project the Neural Image Captioning (NIC) model proposed by Vinyals et al. was reimplemented.

This project was carried out as part of the ID2223 "Scalable Machine Learning and Deep Learning" course at [KTH Royal Institute of Technology](http://kth.se).

### Contributors
- Martin Hwasser (github: [hwaxxer](https://github.com/hwaxxer/)) 
- Wojciech Kryściński (github: [muggin](https://github.com/muggin/))
- Amund Vedal (github: [amundv](https://github.com/amundv))

### References
The implemented architecture was based on the following publication:
- ["Show and Tell: A Neural Image Captiong Generator" by Vinyals et al.](https://arxiv.org/abs/1411.4555)

### Datasets
Experiments were conducted using the [Common Objects in Context](http://cocodataset.org/) dataset. The following subsets were used:
- Training: 2014 Contest Train images [83K images/13GB]
- Validation: 2014 Contest Val images [41K images/6GB]
- Test: 2014 Contest Test images [41K images/6GB]

### Architecture
The NIC architecture consists of two models, the Encoder and a Decoder. The Encoder, which is a Convolutional Neural Network, is used to create a (semantic) summary of the image in a form of a fixed sized vector. The Decoder, which is a Recurrent Neural Network, is used to generate the caption in natural language based on the summary vector created by the encoder.

### Experiments
#### Goals
The goal of the project was to implement and train a NIC architecture and evaluate its performance. A secondary goal, was to check how the type of a recurrent unit and the size of the embeddining in the Decoder (Language Generator) affects the overall performance of the NIC model.

#### Setup
The Encoder was a `ResNet-34` architecture with pre-trained weights on the `ImageNet` dataset. All weights, except from the last layer, were frozen during the training procedure.

The Decoder was a single layer recurrent neural network. The different Recurrent units were tested, `Elman`, `LSTML`, and `GRU`.

Training parameters:
- Learning rate: `1e-3`, with Step LR decay
- Batch size: 128
- Vocabulary size: 15k

#### Evaluation Methods
Experiments were evaluated in a qualitative and quantitative manner. Qualitatitve evluation aimed to assess the coherence of the generated sequences and their relevance given the input image. Quantitative evluation used the following metrics: `BLEU-1`, `BLEU-2`, `BLEU-3`, `BLEU-4`, `ROGUE-L`, `METEOR`, and `CIDEr`. 

### Results
#### Quantitative
- Show scores for each model in table
- Show reference scores from paper

#### Qualitative
#  <div>
#  <img align="center" src="/misc/ss1.png" width=405>
#  <img align="center" src="/misc/ss2.png" width=415>
#  </div>
