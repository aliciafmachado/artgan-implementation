# Implementation of ArtGAN

Project of Deep Learning: Implementation of the neural network ArtGAN

We implement the article presented here: https://arxiv.org/pdf/1702.03410.pdf

Project in colaboration with Iago Martinelli Lopes (Maicken)

## Usage

HOW TO USE:
- First get the dataset Wikiart which is disponible at: http://web.fsktm.um.edu.my/~cschan/source/ICIP2017/wikiart.zip
- Then you must add the files disponible in the folder extra in the Wikiart dataset folder
- After that you must create an environment with conda and install the necessary libraries
- Finally you must run in the command line:
  $ python main.py [artist/genre/style] [version] 
  
Tips:
- You may use aria2 to download the dataset if through command line, since it's faster
- You may also use tmux to leave the process in background

## ArtGAN innovation

The work proposed in the article is an extension of a GAN. It proposes a semi-supervised learning strategy, which consists of allowing backpropagation of the loss function w.r.t. the labels (randomly assigned to each generated images) to the generator from the discriminator. With this feedback, the generator can learn better - in terms of image quality - and faster than a normal GAN.

## Results

Here we give some examples of images produced by the neural network:

### Artist

The images, from left ot right and up to down, correspond to the artists labels Martiros Saryan, Nicholas Roerich, Camille Pissarro, Albrecht Durer, Ivan Aivazovsky and Vincent van Gogh.

![alt text](https://github.com/aliciafmachado/artgan-implementation/blob/master/results/images/artist_grid.png "Artist grid")

Here, we show our Discriminator and Generator loss for 100 epochs of training:

![alt text](https://github.com/aliciafmachado/artgan-implementation/blob/master/results/images/artist_loss.png "Artist loss")

### Genre

Here, we present the results for Genre. The columns of images, from left to right, correspond to the genre labels Portrait, Cityscape and Landscape.

![alt text](https://github.com/aliciafmachado/artgan-implementation/blob/master/results/images/genre_grid.png "Genre grid")

Here, we show our Discriminator and Generator loss for 100 epochs of training:

![alt text](https://github.com/aliciafmachado/artgan-implementation/blob/master/results/images/genre_loss.png "Genre loss")

### Style

These are the results for style. We have that the images, from left to right and up to down, correspond to Baroque, Contemporary Realism, Impressionism,
Naive Art Primitivism, Pop Art and Romanticism. 

![alt text](https://github.com/aliciafmachado/artgan-implementation/blob/master/results/images/style_grid.png "Style grid")

Here, we show our Discriminator and Generator loss for 100 epochs of training:

![alt text](https://github.com/aliciafmachado/artgan-implementation/blob/master/results/images/loss.png "Style loss")

### CIFAR-10

Here we show the results for CIFAR-10. Each image corresponds to a label in the CIFAR-10 dataset:

![alt text](https://github.com/aliciafmachado/artgan-implementation/blob/master/results/images/cifar-10_grid.png "CIFAR-10 grid")

Here, we show our Discriminator and Generator loss for 50 epochs of training:

![alt text](https://github.com/aliciafmachado/artgan-implementation/blob/master/results/images/cifar-10_loss.png "CIFAR-10 loss")
