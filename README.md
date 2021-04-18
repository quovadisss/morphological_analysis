# On the data-driven generation of new service idea: Integrated approach of morphological analysis and text mining

## Project explanation
This project is to create a morphological matrix using a new framework for New Service Development. To accomplish this project, network analysis and clustering were conducted. The news data are crawled in the Fintech and the Healthcare IT industries.

## Framework
1) Crawl Healthcare IT news and Fintech news.
2) Define rules for data preprocessing by building the main elements of a morphological matrix.
3) Collect base keywords for each element.
4) Filter unimportant keywords using structural hole.
5) Cluster keywords using K-Means with a pretrained W2V dictionary.
6) Build a morphological matrix for each industry.

## Usage
1) ```python crawling.py```
2) ```python morphology.py```
3) ```python clustering.py```
