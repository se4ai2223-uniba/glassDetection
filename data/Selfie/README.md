# Selfie Dataset Summary

---
annotations_creators:
- expert-generated
language: []
language_creators:
- other
license:
- apache-2.0
multilinguality: []
pretty_name: Selfie Dataset
size_categories:
- 10K<n<100K
source_datasets:
- original
tags: []
task_categories:
- image-classification
task_ids: []
---

# Dataset Card for Selfie Dataset

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Splits](#data-splits)
- [Dataset Creation](#dataset-creation)
  - [Source Data](#source-data)
  - [Annotations](#annotations)
- [Considerations for Using the Data](#considerations-for-using-the-data)
  - [Known Limitations](#known-limitations)
- [Additional Information](#additional-information)
  - [Dataset Curators](#dataset-curators)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)
  - [Contributions](#contributions)

## Dataset Description

- **Homepage:** https://www.crcv.ucf.edu/data/Selfie/
- **Paper:** https://www.crcv.ucf.edu/data/Selfie/papers/acmmm15/Selfie.pdfs,
- **Point of Contact:** g.dibenedetto39@uniba.it

### Dataset Summary

[Selfie dataset](https://www.crcv.ucf.edu/data/Selfie/) contains 46,836 selfie images annotated with 36 different attributes divided into several categories as follows. Gender: is female. Age: baby, child, teenager, youth, middle age, senior. Race: white, black, asian. Face shape: oval, round, heart. Facial gestures: smiling, frowning, mouth open, tongue out, duck face. Hair color: black, blond, brown, red. Hair shape: curly, straight, braid. Accessories: glasses, sunglasses, lipstick, hat, earphone. Misc.: showing cellphone, using mirror, having braces, partial face. Lighting condition: harsh, dim.



<center>
<img src="./figures/selfie_dataset6.jpg" alt="image_ourCNN" width="600"/>
</center>

### Supported Tasks and Leaderboards

Glass Detection: the dataset can be used to train a model able to identify the presence of glasses in a subject (who take the selfie).


## Dataset Structure

### Data Instances

selfie_dataset.txt provides the annotations and is structured as:
* image name 
* popularity score
* attributes

Below an example image is showed.
<center>
<img src="./figures/sample.jpg" alt="image_ourCNN" width="200"/>
</center>

### Data Fields

[More Information Needed]

### Data Splits

Data has been splitted with the hold-out method. A 80-10-10 split has been done on a subset of Selfie Dataset (25k samples) to create the training set, the validation set and the test set.

|                  | train | validation | test |
|------------------|-------|------------|------|
| Number of images |  20k  | 2,5k       |  2,5k|


## Dataset Creation

### Curation Rationale

[More Information Needed]

### Source Data

#### Initial Data Collection and Normalization

In order to be used for the Glasses Detection task, a phase of feature selection has been done taking in consideration over the 36 attributes just 2 of them: Glasses and Sunglasses. After this phase the dataset has been divided into 2 classes, one with subject wearing Glasses (both Glasses and sunglasses) and the other without. 

With this split the dataset was unbalanced because just the 13% of the samples belong to the Glasses class. In order to overcome this problem an offline data augmentation (flipped, rotation and brightness shifting) phase was performed on the Glasses class, so in this way the dataset becomes balanced.

#### Who are the source language producers?

[More Information Needed]

### Annotations

#### Preprocessing

A preprocessing has been done using the following techniques in the following order:

1. Face alignment: an alignment tool, which is widely used in tasks that work on faces. It can be seen as a form of “data normalization” and it is composed by 3 steps:
    * Center the image
    * Rotate the image such as the eyes lie is on horizontal line
    * Scale the image such that the size of the faces are approximately identical
2. Sharpening filter: used to enhance the edges of objects and adjust the contrast and the shade characteristics.



## Considerations for Using the Data

### Known Limitations

* Unbalanced data with respect to the classes (Glasses and No Glasses) that we are using
* High variability among all the samples in the dataset
* Some images are wrong labeled

## Additional Information

### Dataset Curators

[More Information Needed]

### Licensing Information

[More Information Needed]

### Citation Information

```
@inproceedings{kalayeh2015selfie,
  title={How to Take a Good Selfie?},
  author={Kalayeh, Mahdi M and Seifu, Misrak and LaLanne, Wesna and Shah, Mubarak},
  booktitle={Proceedings of the 23rd Annual ACM Conference on Multimedia Conference},
  pages={923--926},
  year={2015},
  organization={ACM}
}
```



### Contributions

Thanks to [@github-username](https://github.com/<github-username>) for adding this dataset.