# NMT_pytorch
NMT models implemented with pytorch.
For explanations, please see this [blog post](https://petetanru.github.io/projects/char-nmt.html)

The file mainly serves as coding reference. It's in development stage, which means that you have to
open up the code to do configs... (with the risk of tammering other parts of code..!)

#### Running the file
There are several steps you need to do to run the file.

1. From TED Talk 2016, download the desired language pair xml and when building up aligned text, take segments instead of running the entire script to build up sentences. But if you want to use sentences instead, just make sure that Thai is taken as the source language, since it has no EOS punctuation and the script bases its decision on the target language's punctuation.
2. Normalize the punctuation and tokenize your validation set to be used with Moses's BLEU script.

Once that is done, make sure the file path are correct in train.py. This is also the file where you can setup the majority of configurations.
Other than that, simply run the following code in the proper environment.
>python train.py

The easiest way though, which I can't do over github, is to message me and I can share the link to the parallel text that I've built.

#### Dependencies
1. Pytorch
2. Numpy
3. Tensorflow and Keras (this is what deepcut runs on)
4. Deepcut


