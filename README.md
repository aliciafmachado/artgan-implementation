# Implementation of ArtGAN
Project of Deep Learning: Implementation of neural network ArtGAN

We implement the article presented here: https://arxiv.org/pdf/1702.03410.pdf

Project in colaboration with Iago Martinelli Lopes (Maicken)

HOW TO USE:
- First get the dataset Wikiart which is disponible at: http://web.fsktm.um.edu.my/~cschan/source/ICIP2017/wikiart.zip
- Then you must add the files disponible in the folder extra in the Wikiart dataset folder
- After that you must create an environment with conda and install the necessary libraries
- Finally you must run in the command line:
  $ python main.py [artist/genre/style] [version] 
  
Tips:
- You may use aria2 to download the dataset if through command line, since it's faster
- You may also use tmux to leave the process in background

<!---
Results:
We show some of our results:
put grid with style results
put grid with genre results
put grid with style results 
Possible improvements:
Stop discriminator training when it's getting better than the generator
-->
