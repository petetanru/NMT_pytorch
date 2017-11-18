# NMT_pytorch
NMT models implemented with pytorch.
For explanations and reference, please see this [medium blog](https://medium.com/@petepeeradejtanruangporn/experimenting-with-neural-machine-translation-for-thai-1681fd2b375a) or [beta blog post](https://petetanru.github.io/projects/th-nmt.html). 

The file mainly serves as coding reference. It's in development stage, which means that you have to
open up the code to do configs... (with the risk of tammering other parts of code..!)

#### Running the file
There are several steps you need to do before running the code.

**Getting files from TED TALK**
1. From TED Talk 2016, download the desired language pair xml.  
2. When building up aligned text with WIT3's script, read through the perl script and stop at segments instead of running the entire script to build up sentences. If you want to use sentences, just make sure that Thai is processed as the source language, since it has no EOS punctuation and the script bases its decision on the target language's punctuation.

**Evaluation**
1. Download Moses's BLEU, normalizer, and tokenizer script from [here](https://github.com/moses-smt/mosesdecoder). 
2. Normalize and tokenizer your target test script, and make sure to do the same for the text you generate before calculating the score for the two files with the BLEU script. 

**BPE**
1. If you want to do BPE tokenizing, follow the steps [here](https://github.com/rsennrich/subword-nmt) to create a tokenized file. 
2. Once that is done, make sure the file path are correct in train.py and load_data.py. 

**Configuration**
Train.py is where you can setup the majority of configurations, while load_data.py holds the path to your data and calls different preprocessing and loading functions from utils.py. 

Once you have all that sorted out (it can be a big headache, I warn you) run the following code in the proper environment.
>python train.py

The easiest way though, which I can't do over github, is to message me and I can share the link to the parallel text / preprocessed texts that that I've already built. :) 

#### Dependencies
1. Pytorch
2. Numpy
3. Tensorflow and Keras (this is what deepcut runs on)
4. Deepcut
5. pickle

#### References
See the blog post link above!



