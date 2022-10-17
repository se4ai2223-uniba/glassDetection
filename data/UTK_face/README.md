### UTKFace Summary

---
annotations_creators:

- expert-generated
language:
- en
language_creators:
- expert-generated
license:
- apache-2.0
multilinguality:
- monolingual
pretty_name: UTKFace
size_categories:
- 10K<n<100K
source_datasets: []
tags: []
task_categories:
- image-classification
task_ids: []

---

### UTKFace Dataset  Summary

---

---

# Dataset Card for UTKFace

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

- **Homepage**: <https://susanqq.github.io/UTKFace/>

### Dataset Summary

UTKFace dataset is a large-scale face dataset with long age span (range from 0 to 116 years old). The dataset consists of over 20,000 face images with annotations of age, gender, and ethnicity. The images cover large variation in pose, facial expression, illumination, occlusion, resolution, etc.

#### Highlights

- consists of 20k+ face images in the wild (only single face in one image)
- provides the correspondingly aligned and cropped faces
- provides the corresponding landmarks (68 points)
- images are labelled by age, gender, and ethnicity

### Supported Tasks

This dataset could be used on a variety of tasks, e.g., face detection, age estimation, age progression/regression, landmark localization.

- `glass-detection`: the dataset can be used to train a model for Glass-detection, which consists in understanding if a subject into an image wears or not glass. Success on this task is typically measured by achieving a high accuracy. The "GlassDect" (CNN) model currently achieves the **98%** of accuracy.

### Languages

The only supported language by the UTK dataset is english.

## Dataset Structure

### Data Instances

Below an example of images is showed:

<center>

<img src="./figures/samples.png" alt="utk" width="450"/>
</center>

### Data Fields

The images in this dataset cover large pose variations and background clutter. UTKFace has large diversities, large quantities, and rich annotations.

The labels of each face image is embedded in the file name, formated like `age`, `gender`,`race`,`date&time`

- `age` is an integer from 0 to 116, indicating the age
- `gender` is either 0 (male) or 1 (female)
- `race` is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern).
- `date&time` is in the format of `yyyymmddHHMMSSFFF`, showing the date and time an image was collected to UTKFace

### Data Splits

Data has been splitted with the hold-out method. A 80-10-10 split has been done on a subset of the CelebA Dataset (10k samples) to create the training set, the validation set and the test set.

|                  | train | validation | test |
|------------------|-------|------------|------|
| Number of images |  8k   | 1k         |  1k  |

## Dataset Creation

The dataset used for the glass detection has been made by a private group, if you need more information you can reach the [contact](#modified-dataset-curators)

### Curation Rationale

The choice of this dataset is due to two main facts:

- It contains face images, as **[Selfie Dataset](../data/Selfie/README.md)**

- Each image has the glass label, that is essential in a glass detection task

#### Initial Data Collection and Normalization

The private group that has labeled the dataset for the glass detection, has create an .h5py version for practical reasons that contains the images and the related glass label. Considering the two resulting, these were unbalanced, so a balancing phase has been done using the offline data augmentation(flipped, rotation and brightness shifting) only to the images with subject wearing glass.

The dataset has been preprocessed in the following way:

1. Face alignment: an alignment tool, which is widely used in tasks that work on faces. It can be seen as a form of “data normalization” and it is composed by 3 steps:
    - Center the image
    - Rotate the image such as the eyes lie is on horizontal line
    - Scale the image such that the size of the faces are approximately identical
2. Sharpening filter: used to enhance the edges of objects and adjust the contrast and the shade characteristics.

### Personal and Sensitive Information

- Please note that all the images are collected from the Internet which are not property of AICIP. AICIP is not responsible for the content nor the meaning of these images.
- The copyright belongs to the original owners. If any of the images belongs to you, please let us know and we will remove it from our dataset immediately.

## Considerations for Using the Data

### Other Known Limitations for the Original Dataset

The ground truth of age, gender and race are estimated through the DEX algorithm and double checked by a human annotator.

### Other Known Limitations for the Modified Dataset

The ground truth of glasses labels are estimated through a human annotator.

## Additional Information

### Original Dataset Curators

Please contact Yang Song or [Zhifei Zhang](https://zzutk.github.io/) for questions.

### Modified Dataset Curators

For more info contact `g.dibenedetto39@studenti.uniba.it`

### Licensing Information

The UTKFace dataset is available for non-commercial research purposes only.

### Citation Information

```
@inproceedings{zhifei2017cvpr
  title={Age Progression/Regression by Conditional Adversarial Autoencoder},
  author={Zhang, Zhifei, Song, Yang, and Qi, Hairong},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2017},
  organization={IEEE}
}
```
