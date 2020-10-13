# Models used by COVID PneumoCheck App

___Models converted and forked from the COVID-Net Open Source Initiative (CNOSI)___

__A fork of the COVID-Net Open Source Initiative to convert and prepare the models for mobile development.__

## Get Started

To get started using a TFlite model in an android application consider the following examples:

[Tensorflow Android Example](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android)

[Live Image Recognition Article by Ahmed Gad](https://heartbeat.fritz.ai/image-recognition-for-android-with-a-custom-tensorflow-lite-model-6418186ecc0e)


## Two Files are needed for Tensorflow Lite Android Mobile Usage:
1. The list of labels - Labels.txt
	
2. The TFlite model - model.tflite

<br>

The labels.txt file will be used to label the class names (types), order is important :
[labels.txt](https://drive.google.com/file/d/1Bl0i_0D805eDT7YMzRUwB2NUU4R_Ljw1/view?usp=sharing)


### TFLite Version for mobile - COVIDNet Chest X-Ray Classification

_Currently I have converted the CXR4 models A and B to TFLite with no optimisation, Default Optimisation and 16Float Optimisation._

|  Type  | Input Resolution | COVID-19 Sensitivity | Optimisation | Size |       Model      |
|:------:|:----------------:|:--------------------:|:------------:|:----:|:----------------:|
| TFlite |      480x480     |         ?            |   None       | 153M | [covidnet_a_unoptimised.tflite](https://drive.google.com/file/d/1skRubZENnJ6E0deTvsTZs0j3fnMV-Qld/view?usp=sharing)|
| TFlite |      480x480     |         95.0         |   Default    |  40M | [covidnet_a.tflite](https://drive.google.com/file/d/1_DWDkJgFnP_EtvWMMA4FdZBvxLj48T-y/view?usp=sharing)|
| TFlite |      480x480     |         93.0         |   Default    |  12M | [covidnet_b.tflite](https://drive.google.com/file/d/1lUQfmPN1KLXBkGfmPUejFCsAP10zWqkQ/view?usp=sharing)|
| TFlite |      480x480     |         ?            |   16Float    |  81M | [converted_model_a_16floatoptim.tflite](https://drive.google.com/file/d/1f0s07L7QXbLyEnAc2bCc9JM67I58JA_T/view?usp=sharing)|
| TFlite |      480x480     |         ?            |   16Float    |  23M | [converted_model_b_16floatoptim.tflite](https://drive.google.com/file/d/1G7MDML2b9iUT-lm30ulv9sgahyToRFB-/view?usp=sharing)|

<br>

These models were converted from the following checkpoint unfrozen graph models:

<br>

#### ORIGINAL COVIDNet Chest X-Ray Classification
|  Type | Input Resolution | COVID-19 Sensitivity | Accuracy | # Params (M) | MACs (G) |        Model        |
|:-----:|:----------------:|:--------------------:|:--------:|:------------:|:--------:|:-------------------:|
|  ckpt |      480x480     |         95.0         |   94.3   |      40.2    |  23.63   |[COVIDNet-CXR4-A](https://bit.ly/COVIDNet-CXR4-A)|
|  ckpt |      480x480     |         93.0         |   93.7   |      11.7    |   7.50   |[COVIDNet-CXR4-B](https://bit.ly/COVIDNet-CXR4-B)|

<br><Br>


### For inference using the TFLite models on python environment:

**Use Tensorflow v 1.15.3**

<br>

## Steps for inference (classifying) in a Python Environment for trialing the models
**DISCLAIMER: Do not use this prediction for self-diagnosis. You should check with your local authorities for the latest advice on seeking medical assistance.**

1. Download a TFLite model (e.g. "covidnet_a.tflite") and the 'labels.txt'
2. Locate models and xray image to be inferenced
3. To inference,
```
python3 inference-tflite.py --model_file
  /Users/<username>/Desktop/AndroidCovidNet/TFLITEModels/covidnet_b.tflite --label_file
  /Users/<username>/Desktop/AndroidCovidNet/TFLITEModels/labels.txt --image
  /Users/<username>/Desktop/AndroidCovidNet/images/pneu1115.jpg
```
Output will display on the terminal console.


## Converting from CovidNet to TFlite

To create your own TFlite models from the Models provided by the Core COVID-Net Team a few key steps are required:

1. Freeze the graph and export the frozen graph converting it from v1 tensorflow graph to a frozen graph .pb format
2. Convert the frozen graph model to tflite model

The CovidNetModel2Tflite.py can help perform these two steps.

For more options and information, `python3 CovidNetModel2Tflite.py --help`


#### About the CovidNet Models


COVIDNet-CXR4 models takes as input an image of shape (N, 480, 480, 3) and outputs the softmax probabilities as (N, 3), where N is the number of batches. If using the TF checkpoints, here are some useful tensors:

- 	**input tensor: input_1:0**

	- Input tensor name is "input_1"

-	logit tensor: norm_dense_1/MatMul:0

-	**output tensor: norm_dense_1/Softmax:0**

	- Output tensor name is "norm_dense_1/Softmax"

-	label tensor: norm_dense_1_target:0

-	class weights tensor: norm_dense_1_sample_weights:0

-	loss tensor: loss/mul:0



For inference using the Core COVID-Net models (Unfrozen Graphs) (making predictions on an images):

## Steps for inference
**DISCLAIMER: Do not use this prediction for self-diagnosis. You should check with your local authorities for the latest advice on seeking medical assistance.**

1. Download a model from the [pretrained models section](models.md)
2. Locate models and xray image to be inferenced
3. To inference,
```
python inference.py \
    --weightspath models/COVIDNet-CXR3-B \
    --metaname model.meta \
    --ckptname model-1014 \
    --imagepath assets/ex-covid.jpeg
```
4. For more options and information, `python inference.py --help`

<br><br><Br>

### COVID PneumoCheck team
* Danny Falero

<br><br><Br>
<hr>

### From Core COVID-Net Team, for more view [their README](README.original.md)

Please consider CNOSI's disclaimer as you utilise the models and resources found in this repository.

**Note: The COVID-Net models provided here are intended to be used as reference models that can be built upon and enhanced as new data becomes available. They are currently at a research stage and not yet intended as production-ready models (not meant for direct clinical diagnosis), and we are working continuously to improve them as new data becomes available. Please do not use COVID-Net for self-diagnosis and seek help from your local health authorities.**

**Recording to webinar on [How we built COVID-Net in 7 days with Gensynth](https://darwinai.news/fny)**

**Update 07/08/2020:** We released COVIDNet-CT, which was trained and tested on 104,009 CT images from 1,489 patients. For more information, as well as instructions to run and download the models, refer to [this repo](https://github.com/haydengunraj/COVIDNet-CT).\
**Update 06/26/2020:** We released 3 new models, COVIDNet-CXR4-A, COVIDNet-CXR4-B, COVIDNet-CXR4-C, which were trained on the new COVIDx4 dataset with over 14000 CXR images and 473 positive COVID-19 images for training. The test results are based on the same test dataset as COVIDNet-CXR3 models.\
**Update 06/01/2020:** We released an [inference script](docs/covidnet_severity.md) and the [models](docs/models.md) for  geographic and opacity extent scoring of SARS-CoV-2 lung severity.\
**Update 05/26/2020:** For a detailed description of the methodology behind COVID-Net based deep neural networks for geographic extent and opacity extent scoring of chest X-rays for SARS-CoV-2 lung disease severity, see the paper [here](https://arxiv.org/abs/2005.12855).\
**Update 05/13/2020:** We released 3 new models, COVIDNet-CXR3-A, COVIDNet-CXR3-B, COVIDNet-CXR3-C, which were trained on a new COVIDx dataset with both PA and AP X-Rays. The results are now based on a test set containing 100 COVID-19 samples.\
**Update 04/16/2020:** If you have questions, please check the new [FAQ](docs/FAQ.md) page first.

**By no means a production-ready solution**, the hope is that the open access COVID-Net, along with the description on constructing the open source COVIDx dataset, will be leveraged and build upon by both researchers and citizen data scientists alike to accelerate the development of highly accurate yet practical deep learning solutions for detecting COVID-19 cases and accelerate treatment of those who need it the most.

Currently, the COVID-Net team is working on **COVID-RiskNet**, a deep neural network tailored for COVID-19 risk stratification.  Currently this is available as a work-in-progress via included `train_risknet.py` script, help to contribute data and we can improve this tool.

If you would like to **contribute COVID-19 x-ray images**, please submit to https://figure1.typeform.com/to/lLrHwv. Lets all work together to stop the spread of COVID-19!

Our desire is to encourage broad adoption and contribution to this project. Accordingly this project has been licensed under the GNU Affero General Public License 3.0. Please see [license file](LICENSE.md) for terms. If you would like to discuss alternative licensing models, please reach out to us at linda.wang513@gmail.com and a28wong@uwaterloo.ca or alex@darwinai.ca


If there are any technical questions after the README, FAQ, and past/current issues have been read, please post an issue or contact:
* desmond.zq.lin@gmail.com
* paul@darwinai.ca
* jamesrenhoulee@gmail.com
* linda.wang513@gmail.com
* ashkan.ebadi@nrc-cnrc.gc.ca

If you find their work useful, can cite their paper using:

```
@misc{wang2020covidnet,
    title={COVID-Net: A Tailored Deep Convolutional Neural Network Design for Detection of COVID-19 Cases from Chest Radiography Images},
    author={Linda Wang, Zhong Qiu Lin and Alexander Wong},
    year={2020},
    eprint={2003.09871},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Core COVID-Net Team
* DarwinAI Corp., Canada and Vision and Image Processing Research Group, University of Waterloo, Canada
	* Linda Wang
	* Alexander Wong
	* Zhong Qiu Lin
	* Paul McInnis
	* Audrey Chung
	* Hayden Gunraj, COVIDNet for CT: https://github.com/haydengunraj/COVIDNet-CT
* Vision and Image Processing Research Group, University of Waterloo, Canada
	* James Lee
* Matt Ross and Blake VanBerlo (City of London), COVID-19 Chest X-Ray Model: https://github.com/aildnont/covid-cxr
* Ashkan Ebadi (National Research Council Canada)
* Kim-Ann Git (Selayang Hospital)
* Abdul Al-Haimi, COVID-19 ShuffleNet Chest X-Ray Model: https://github.com/aalhaimi/covid-net-cxr-shuffle

## Table of Contents
1. [Requirements](#requirements) to install on your system
2. How to [generate COVIDx dataset](docs/COVIDx.md)
3. Steps for [training, evaluation and inference](docs/train_eval_inference.md) of COVIDNet
4. Steps for [inference](docs/covidnet_severity.md) of COVIDNet lung severity scoring
5. [Results](#results)
6. [Links to pretrained models](docs/models.md)

## Requirements

The main requirements are listed below:

* Tested with Tensorflow 1.13 and 1.15
* OpenCV 4.2.0
* Python 3.6
* Numpy
* Scikit-Learn
* Matplotlib
* Tensorflow Lite
