# Action-Recognition-in-the-Wild

# Execution Flow


## Dataset preparation 

1.	Acquire data set  [UCF 101](https://www.crcv.ucf.edu/data/UCF101.php)
2.	Update the 'dataPath' variable 
3.	Run the dataset preparation file using python3.
4.  Now you'll have rames and optical flow coorresponding to each video.


## Testing and Training:

Before executing the testing and training steps please refer to [installation](https://www.tensorflow.org/install) guide and install the required modules.
The 2 models (Two Stream and Encoder-Decoder ) can be executed using the files provided under the Models folder.
## Two Stream
After preparing the data, open Model_two_stream file found in the Model folder.
## Encoder-Decoder
After preparing the data, open Model_ED file found in the Model folder.
## Prediction
The predictions can be made by running the try and try_ed file.

## OUTPUT
After running all of the above mentioned scripts, you will have a final average training accuracy at the end which will be of that models

