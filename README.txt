September 2021

Code written by:
Brittny Martinez (Email: bmart772 (at) gmail (dot) com)
Hyunjoung Yang (Tagged in repo as Hyun-Y; Email: hyunjoung (dot) yang (at) gmail (dot) com)


The purpose of this repo is to create a Perceptron learning model and a model utilizing the
WINNOW algorithm and test it against the Mushroom Dataset found in the UCI repository with and without
class label noise.

IMPLEMENTATION NOTES:
We used One-Hot-Encoding and class label encoding to convert discrete values into binary values. 
Though the class labels were naturally binary, the attributes were multiplied from 22 attributes
to about 100-120 attributes as each possible value in each attribute was converted to a binary
attribute itself due to the One-Hot-Encoding.

STATS:

WITHOUT CLASS LABEL NOISE
WINNOW Algorithm Model - 100% Accuracy and maintained it through the 5 EPOCHS
Perceptron Model - Started at 98% Accuracy at EPOCH 1 and achieved 100% Accuracy by EPOCH 5

WITH CLASS LABEL NOISE
WINNOW Algorithm Model - Suffered greatly from class label noise.
FOR WINNOW:
		Noise %      Accuracy 
		 0%          100%
		10%          ~85%
		20%          ~68%
        30%          ~56%
		40%          ~53%
		50%          ~50%       Which mkes sense

Perceptron model - Suffered from class label noise as well but to a lesser extent
FOR Perceptron:
		Noise %    Accuracy
		 0%        100%
		10%        ~89%
		20%        ~78%
		30%        ~66%
		40%        ~58%
		50%        ~50%        Which, of course, makes sense
		
		
Conclusion:
The WINNOW model seems like a great method to help with attribute based noise but suffers greatly
when introducing class label noise.
The Perceptron model is the opposite. The Perceptron model does well in the face of class label noise
but suffers greatly from attribute value noise.
