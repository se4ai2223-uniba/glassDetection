### CelebFaces Dataset (CelebA) Summary

---
annotations_creators:

- expert-generated
language:
- en
language_creators:
- crowdsourced
license:
- apache-2.0
multilinguality:
- monolingual
pretty_name: CelebA
size_categories:
- 100K<n<1M
source_datasets:
- original
tags: []
task_categories:
- image-classification
task_ids: []

---

# Dataset Card for CelebA

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Supported Tasks](#supported-tasks-and-leaderboards)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
- [Dataset Creation](#dataset-creation)
  - [Curation Rationale](#curation-rationale)
  - [Initial Data Collection and Normalization](#initial-data-collection-and-normalization)
  - [Personal and Sensitive Information](#personal-and-sensitive-information)
- [Considerations for Using the Data](#considerations-for-using-the-data)
  - [Social Impact of Dataset](#social-impact-of-dataset)
  - [Other Known Limitations](#other-known-limitations)
- [Additional Information](#additional-information)
  - [Dataset Curators](#dataset-curators)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)
  - [Contributions](#contributions)

## Dataset Description

- **Homepage**: <https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>

### Dataset Summary

CelebFaces Attributes dataset contains 202,599 face images of the size 178×218 from 10,177 celebrities, each annotated with 40 binary labels indicating facial attributes like hair color, gender and age.

### Supported Tasks

The dataset can be employed as the training and test sets for the following computer vision tasks: face attribute recognition, face recognition, face detection, landmark (or facial part) localization, and face editing & synthesis.

- glass-detection: The dataset can be used to train a model for Glass-detection, which consists in understanding if a subject into an image wears or not glass. Success on this task is typically measured by achieving a high accuracy. The "GlassDect" (CNN) model currently achieves the **97%** of accuracy.

### Languages

The only supported language by the CelebA dataset is english.

## Dataset Structure

### Data Instances

**[list_attr_celeba.txt](./raw/list_attr_celeba.txt)** provides the annotations and is structured as:
- image name
- attributes

Below an example of images is showed:
<center>

<img src="./figures/exampleCelebA.png" alt="CelebA" width="450"/>
</center>

### Data Fields

The images in this dataset cover large pose variations and background clutter. CelebA has large diversities, large quantities, and rich annotations, including:

- 10,177 number of identities
- 202,599 number of face images
- 5 landmark locations, 40 binary attributes annotations per image.

The attributes present for each sample are: 5_o_Clock_Shadow, Arched_Eyebrows, Attractive, Bags_Under_Eyes, Bald, Bangs, Big_Lips, Big_Nose, Black_Hair, Blond_Hair, Blurry, Brown_Hair, Bushy_Eyebrows, Chubby, Double_Chin, Eyeglasses, Goatee, Gray_Hair, Heavy_Makeup, High_Cheekbones, Male, Mouth_Slightly_Open, Mustache, Narrow_Eyes, No_Beard, Oval_Face, Pale_Skin, Pointy_Nose, Receding_Hairline, Rosy_Cheeks, Sideburns, Smiling, Straight_Hair, Wavy_Hair, Wearing_Earrings, Wearing_Hat, Wearing_Lipstick, Wearing_Necklace, Wearing_Necktie, Young.

### Data Splits

Data has been splitted with the hold-out method. A 80-10-10 split has been done on a subset of the CelebA Dataset (10k samples) to create the training set, the validation set and the test set.

|                  | train | validation | test |
|------------------|-------|------------|------|
| Number of images |  8k   | 1k         |  1k  |

## Dataset Creation

### Curation Rationale

The choice of this dataset is due to two main facts:
- It contains face images, as **[Selfie Dataset](../data/Selfie/README.md)**

- Each image has the glass label, that is essential in a glass detection task

#### Initial Data Collection and Normalization

From the original data collection, a `.h5` file has been created for practical reasons to be used in the training phase. [comment](come è struttrato) Considering only the glass label, the two resulting classes were unbalanced, so a balancing phase has been done using the offline data augmentation only to the images with subject wearing glass.

The dataset has been preprocessed in the following way:

1. Face alignment: an alignment tool, which is widely used in tasks that work on faces. It can be seen as a form of “data normalization” and it is composed by 3 steps:
    - Center the image
    - Rotate the image such as the eyes lie is on horizontal line
    - Scale the image such that the size of the faces are approximately identical
2. Sharpening filter: used to enhance the edges of objects and adjust the contrast and the shade characteristics.

### Personal and Sensitive Information

The face identities are released upon request for research purposes only.

## Considerations for Using the Data

### Social Impact of Dataset

[More Information Needed]

### Discussion of Biases

[More Information Needed]

### Other Known Limitations

- You agree not to reproduce, duplicate, copy, sell, trade, resell or exploit for any commercial purposes, any portion of the images and any portion of derived data.
- You agree not to further copy, publish or distribute any portion of the CelebA dataset. Except, for internal use at a single site within the same organization it is allowed to make copies of the dataset.

## Additional Information

### Dataset Curators

Please contact [Ziwei Liu](https://liuziwei7.github.io/) and Ping Luo for questions about the dataset.

### Licensing Information

The CelebA dataset is available for non-commercial research purposes only.

### Citation Information

```
@inproceedings{liu2015faceattributes,
  title = {Deep Learning Face Attributes in the Wild},
  author = {Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
  booktitle = {Proceedings of International Conference on Computer Vision (ICCV)},
  month = {December},
  year = {2015} 
}
```
