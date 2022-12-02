# Document-AI-2022-Wrap-up

# Document-AI-2022
[![Build status](https://ci.appveyor.com/api/projects/status/miah0ikfsf0j3819/branch/master?svg=true)](https://ci.appveyor.com/project/zdenop/tesseract/)
[![GitHub license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://raw.githubusercontent.com/tesseract-ocr/tesseract/main/LICENSE)
[![Downloads](https://img.shields.io/badge/download-all%20releases-brightgreen.svg)](https://github.com/tesseract-ocr/tesseract/releases/)
[![CodeFactor](https://www.codefactor.io/repository/github/elvinagam/tesseract-ocr-aze/badge)](https://www.codefactor.io/repository/github/elvinagam/tesseract-ocr-aze)


This notebook goes through how usual Document AI pipeline works, includes details about SOTA models and papers in the field, also some details regarding how in H2O.ai Document AI team we use it. Quick heads-up, this notebook is mostly for the purpose of quickly going through the doc ai field and briefly looking at the most recent works, their usage methods, how different they are to the current SOTA, license, etc. If further information is needed, please check the papers listed below.

Table of Contents
=================

* [Document-AI-2022](#Document-AI-2022)
   * [About](#about)
   * [License](#license)
   * [Intro](#intro)
   * [Document-AI-argus](#document-ai-argus)
   * [Document-Image-Token-Classification-Task](#document-image-token-classification-task)
   * [LayoutLM Ancestors](#layoutlm-ancestors)
   * [Drawbacks of Ancestors](#drawbacks-of-ancestors)
   * [LayoutLMv1](#layoutlmv1)
   * [Overview](#overview)
   * [Setup ](#layoutlmv1)
   * [Fine-tuning](fine-tuning)
   * [LayoutLMv2](#layoutlmv2)
   * [Overview](#overview)
   * [Training-and-Inference](#training-and-inference)
   * [Sequence-of-actions-under-the-hood](#sequence-of-actions-under-the-hood)
   * [LayoutLMv3](#layoutlmv3)
   * [Overview](#overview)
   * [Training-and-Inference](#training-and-inference)
   * [v1-v2-v3](#v1-v2-v3)
   * [Donut-LiLT](#donut-lilt)
   * [Licensing](#licensing)
   * [Datasets](#datasets)
   * [Papers](#papers)
   * [Refs](#refs)
   

# Intro
## Document-AI-Argus

Document AI usually consists of several stages in a complex pipeline starting from preprocessing, OCR-ing, labeling, all the way up to detection, recognition, (rotation), training, inference and ending with postprocessing. This notebook does not cover preprocessing and OCR steps, rather focuses on training procedure - which can be considered fine-tuning in our case.

Overall idea of document ai is to take documents (in different formats such as, PDF, word, TIFF, etc) as input, and be able to identify corresponding data of interest. Usually pdf documents can be in 2 ways.

PDFs with embedded text
PDFs with non-embedded text
Most common scenario is having both cases in a single pdf document. So, most complete pipeline is:

1. PDF Handling - After extracting the embedded text from the PDF file and saving it to the VIA file, each page is converted into PNG format

2. OCR - each PNG file goes through OCR - in our case, DocTR which handles images, finds texts and corresponding tokens and saves it to the VIA file

3. Labeling - for labeling, VGG Viewer is used which outputs labels, and bounding boxes for the targets that we are looking for

4. Merge OCR + Labels - Right before the training, we merge the VIA files we have, which combines OCR tokens with labels and bounding boxes (coordinates)

5. LayoutLM - LayoutLMv1 takes its own format for the training procedure and we train the LayoutLMv1 model againsts all the PDFs we had which had been labeled.

6. Inference

7. Postprocessing

## Document-Image-Token-Classification-Task

Main task here consists of 2 parts.

* First is detecting the right target in a document, recognizing it correctly which completes the OCR process

* Second is classfying those found targets into which labels they belong to which completes the LayoutLM process.
For the first stage, they are several OCR models like DocTR (which we use), EasyOCR, PaddlePaddle, TrOCR, etc.

For the second stage, previously, BERT-like models were mainly used, However, considering the fact that BERT does not consider layout and visual information when classifying tokens, it was not performing really well. Right after adding the functionality of layout, visual information and both of these together, DocAI benchmarks currently sees 95% accuracy.

On RVL-CDIP benchmarks:

> BERT using only textual information - 89%

> DiT using only vision input, without text - 92%

> LayoutLM v3 using both layout and visual input - 95%

Considering LayoutLM and DiT runs the state-of-the-art, let's see what backbones does they stand with.

## LayoutLM Ancestors

In this section, details included are not necessarily used in argus currently, however, explains how the LayoutLM versions came up to be and how they relate to its different ancestors.

As mentioned above, most of the SOTA models are based on Mask Regional CNNs originally. However, where does Mask R-CNN itself come from and why?

### CNN
Starting of with the building blocks, CNNs are obviously the main bulding blocks which help to work with pixels and create a feature map to identify objects on it. However, if there are multiple objects of interest on images, we will have to divide the images into thousands of smaller regions to find the object we are looking for. This is computationally expensive on lots of real-world scenarios, thus, a better version of CNNs is suggested which is R-CNNs

### R-CNN - Region based Convolutions
Previous problem was about having thousands of regions of interest to go through. R-CNN paper suggests downsizing it to only 2000 region proposals and only feeding those into network. Out of thousands of pixels, 2000 region proposals are selected by using selective search algorithm. Procedure is as follows:

1. Select 2000 image proposals for an image through selective search.
2. Warp image regions to be the same size before feeding to CNN
3. Feed image proposals to CNN
4. Classify each image (SVM) proposals with Support Vector Machines about the presence of target object
5. Add bounding boxes to each proposal through Bounding Box Regressor
6. Takes 20 seconds

#### Primary challenge
Classifying 2k images proposals for one image still takes quite a lot of time. It takes around 20 seconds or more which means it is not real-time. And lastly, selective search algorithm is a greedy search algorithm with no loss function. Meaning it does not learn anything from the previous experiences and sometimes lead to generation of bad region proposals

### Fast R-CNN - Fast Region-based Convolutions 
This time, the previous R-CNN version is improved and the drawbacks are handled so that the algorithm would be 10 times faster. This time the procedure and sequence are changed. Instead of creating region proposals, we let CNNs to decide where to look and create the feature map. From the feature map, we infer region of interests and feed into FC layer.

1. Instead of feeding 2000 proposals into CNN, feed whole image to CNN to generate region proposals
2. ROI pooling
3. Fully connected layer
4. Softmax and bbox regressor
5. Takes 2 seconds
#### Primary challenge 
We still use the selective search algorithm right after we get the feature map from CNNs to decide which regions of interest we should select. Selective search stays to be the main bottleneck of Fast R-CNN untill Faster R-CNN paper is proposed.

### Faster R-CNN - Faster Regional CNN
This time, rather than using bold greedy algorithm which does not learn through its selection journey, a Region Proposal network is added to the model which boosts the performance to be 10 times faster.

1. Solves region proposal bottleneck

2. Add a network for choosing region of interest - Regional Proposal Network

3. Region Proposal Network tells NN where to look - Attention

4. Takes 0.2 seconds

#### Primary challenge
This is not a challenge but a missing feature which is, Faster R-CNN does not have semantic segmentation. While it gives bounding boxes for a cat, it does not specifically draw lines on where the cat is located inside picture.

### Mask R-CNN Masked Regional CNNs
This network adds the feature of image segmentation and instance segmentation. In addition to the regular Faster CNN output, we also add binary mask for each region of interest.

1. In addition to label and bounding box, Mask also outputs object mask
2. Uses ROIAlign to do down to pixel level segmentation
3. Region ProposalNetwork + Binary Mask Classifier
Currently, Mask R-CNNs also have different versions where it is improved to handle few edge case scenarios like in this paper.

![Image](https://github.com/elvinagam/document-ai-2022-wrap-up/blob/main/src/rcnn.png)
________________________________________________________________________________________________________________________________


## Drawbacks of Ancestors

All of the models mentioned above are great papers on which the most recent papers rely on. However, there are still few missing pieces which are:

1. They rely on human-labeled documents which are quite few and haven't been trained on large-scale unlabeled data
2. They usually depend on pretrained CNN models rather than training the model from scratch on both textual and visual information


## LayoutLMv1

### Overview

LayoutLM - As mentioned previously, the answer to WHY LAYOUTLM question is the fact that it does not only consider textual information when deciding on token classes, but also it considers all 3 available information

* Text
* Image
* Layout/location
Original LayoutLM model uses Tesseract as its OCR model. However, we as Document AI team uses DocTR OCR model which is not covered in this notebook. Trained on

LayoutLM is pre-trained on the IIT-CDIP Test Collection of Tobacco documents, which contains more than 6 million scanned documents with 11 million scanned images.

Tested on:

3 downstream tasks and datasets:

1. FUNSD dataset for spatial layout analysis and form understanding - it is a noisy dataset collection of 199 fully human annotated forms
2. SROIE dataset for scanned receipt information - it is a dataset collection of 1000 scanned receipt images: .jpg file of the scanned receipt, a .txt file holding OCR information and a .txt file holding the key information values.
3. RVL-CDIP - The RVL-CDIP (Ryerson Vision Lab Complex Document Information Processing) dataset consists of 400,000 grayscale images in 16 classes, with 25,000 images per class. There are 320,000 training images, 40,000 validation images, and 40,000 test images. The images are sized so their largest dimension does not exceed 1000 pixels.

LayoutLM v1 model comes with language embeddings only. That means, final layers with visual embeddings is not shared initially. As it will be discussed in next v2 part, LayoutLM v2 comes with visual embeddings too which uses Detectronv2 library.


### Fine-tuning

Even though LayoutLM is pretrained and still does a good job of recognizing and classifying tokens, it still needs to be fine-tuned based on the business use case that we use it on. It is usually recommended to be fine-tuned at least 200 or more images. LayoutLM needs:

1. OCR bounding boxes for each token and their corresponding cooredinates
2. Classes for each tokens
3. All these merged into layoutlm format

## LayoutLMv2

### Overview
Added in LayoutLMv2

1. LayoutLMv2 - Visual embeddings are added to the pre-training stage unlike v1 where it was on fine-tuning stage

2. Text + Layout + Visual information in pretraining stage

3. In addition to Masked Language Modeling, Text Alignment and Text Image Matching for better alignment of texts
   
### Training-and-Inference

* LayoutLMv2 expects the input images to be resized and normalized. Color channels needs to be in the format of BGR as it was initially used in Detectron2 detection backbone

* Uses Word-Piece tokenization technique

* Each image is expected to be in the size of 224x224 with 3 channels in the form of torch tensor. e.g. would be (batch_size, 3, 224, 224)

* Bounding box inputs can be retrived from OCR tools like Tesseract which outputs 4 cordinates of the word box in the format of (x0, y0, x1, y1). Each corrdinate is normalized to be in the range of 0 to 1000


### Sequence-of-actions-under-the-hood

PDFS can be given to the LayoutLMv2Processor which applies Tesseract OCR engine as a default OCR option. One can set apply_ocr to False to apply custom OCR back-end. Processor has 2 stages:

* LayoutLMv2FeatureExtractor - Applies OCR, gets (a) normalized boundings boxes alongside with the (b) list of words and (c) resized images

* LayoutLMv2Tokenizer - Tokenizes words and bounding boxes into input_ids, attention_mask, token_type_ids, bbox

## LayoutLMv3

### Overview

LayoutLMv3 is getting >90% F1 on FUNSD dataset.

One improvement from LayoutLmv1 was the usage of segment-cell level position embeddings rather than word level position embeddings. OCR engines like Tesseract can already handle that. So, simple tweak and training on those OCR results lead to better results.

### Training-and-Inference

* Different from v2, here color channels are expected to be in usual RBG format

* Byte-level Byte-Pair encoding tokenization is used

## v1-v2-v3

![Benchmark](https://github.com/elvinagam/document-ai-2022-wrap-up/blob/main/src/benchmark.png)


## Donut-LiLT

## Licensing

Even though there are tons of open-source Document AI products out there, as a commercial company, you might not like the licenses of the best architechtures. Currently, only LayoutLmv1 can be used for commercial purposes. Not to forget the fact that Donut and LiLT models are quite powerful and can be used anywhere.

```
|   Model   |    License           |
|                                  |
| LayoutLM  |    MIT               |
| LayoutLMv2|    CC BY-NC-SA 4.0   |
| LayoutXLM |    CC BY-NC-SA 4.0   |
| LayoutLMv3|    CC BY-NC-SA 4.0   |
| Donut     |    MIT               |
| DiT       |    CC BY-NC-SA 4.0   |
| LiLT      |    MIT               |

```

## Datasets


| task | typical metrics | benchmark datasets |
| --- | --- | --- |
| Optical Character Recognition | Character Error Rate (CER) |  |
| Document Image Classification | Accuracy, F1 | [RVL-CDIP](https://huggingface.co/datasets/rvl_cdip) |
| Document layout analysis | mAP (mean average precision) | [PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet), [XFUND](https://github.com/doc-analysis/XFUND)(Forms) |
| Document parsing | Accuracy, F1 | [FUNSD](https://guillaumejaume.github.io/FUNSD/), [SROIE](https://huggingface.co/datasets/darentang/sroie/), [CORD](https://github.com/clovaai/cord) |
| Table Detection and Extraction | mAP (mean average precision) | [PubTables-1M](https://arxiv.org/abs/2110.00061) |
| Document visual question answering | Average Normalized Levenshtein Similarity (ANLS) | [DocVQA](https://rrc.cvc.uab.es/?ch=17) |


## Papers

[TrOCR](https://arxiv.org/abs/2109.10282)

[PaddleOCR](https://arxiv.org/abs/2009.09941)

[EasyOCR](https://huggingface.co/spaces/tomofi/EasyOCR)

[Mask R-CNN](https://arxiv.org/abs/1703.06870)

[LayoutLM](https://arxiv.org/abs/1912.13318)

[LayoutLMv2](https://arxiv.org/abs/2012.14740)

[LayoutXLM](https://arxiv.org/abs/2104.08836)

[LayoutLMv3](https://arxiv.org/abs/2204.08387)

[ERNIE-Layout](https://arxiv.org/abs/2210.06155)

[LiLT](https://arxiv.org/abs/2202.13669)

[TableFormer](https://arxiv.org/abs/2202.13669)

[DETR](https://arxiv.org/abs/2005.12872)

[Donut](https://arxiv.org/abs/2111.15664)

[DiT](https://arxiv.org/abs/2203.02378)

## Refs

[Document AI Transformers](https://github.com/philschmid/document-ai-transformers)

[Transformers Tutorials](https://github.com/NielsRogge/Transformers-Tutorials)

[Accelerating Document AI](https://huggingface.co/blog/document-ai) 


## About

Quick DocAI Overview from past 20 years to 2022 with detailed notes on papers:

![img](https://github.com/elvinagam/document-ai-2022-wrap-up/blob/main/src/lylm.PNG)

## License

    The code in this repository is licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.



