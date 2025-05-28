# V2A Detection based on SlowFast

This repository contains a modified version of the [SlowFast](https://github.com/facebookresearch/SlowFast) video recognition framework, adapted for classification tasks on [The Greatest Hits Dataset](https://andrewowens.com/vis/).

## Overview

The project aims to build a video classification model tailored to sound event detection from visual inputs. It leverages the dual-pathway architecture of SlowFast to extract both high-frequency motion features and low-frequency semantic features from video data. We have adapted and extended the model to suit the unique characteristics of the Greatest Hits dataset.

## Dataset

We use [The Greatest Hits dataset](https://andrewowens.com/vis/), which contains short video clips annotated with sound event labels, allowing supervised training for vision-based audio prediction.

## Base Framework

This code is developed on top of the official [SlowFast repository](https://github.com/facebookresearch/SlowFast) released by Facebook AI Research. Significant modifications have been made to support custom datasets and classification heads tailored for frame-level binary classification tasks.

## Features

- Dual-path SlowFast architecture
- Custom classification head for event detection
- Support for frame-level annotations and sliding-window inputs
- Evaluation metrics customized for framewise prediction

## Installation

Please refer to the [SlowFast installation instructions](https://github.com/facebookresearch/SlowFast/blob/main/INSTALL.md), and ensure the following dependencies are installed:
- PyTorch
- torchvision
- fvcore
- simplejson

## Getting Started

```bash
# Clone the repository
git clone https://github.com/Mavi-sly/v2a_detection.git
cd v2a_detection

# Prepare the dataset and config file in [Greathits](https://github.com/Mavi-sly/v2a_detection/blob/main/configs/GreatHits/greathits.yaml)
# Then start training
python tools/run_net.py --cfg configs/GreatHits/greathits.yaml
