# Implementation of ArtGAN
Project of Deep Learning: Implementation of neural network ArtGAN

We implement the article presented here: https://arxiv.org/pdf/1702.03410.pdf

Project in colaboration with Iago Martinelli Lopes (maicken)

HOW TO USE:
- First get the dataset Wikiart which is disponible at: http://web.fsktm.um.edu.my/~cschan/source/ICIP2017/wikiart.zip
- Then you must add the files disponible in the folder extra in the Wikiart dataset folder
- After that you must create an environment with conda and run the script prep.py
- Finally you must run in the command line:
  $ python main.py [artist/genre/style] [version] 
  
Tips:
- You may use aria2 to download the dataset if through command line, since it's faster
- You may also use tmux to leave the process in background
