# MKR

This repository is the implementation of MKR ([arXiv](https://arxiv.org/abs/1901.08907)):

> Multi-Task Feature Learning for Knowledge Graph Enhanced Recommendation  
Hongwei Wang, Fuzheng Zhang, Miao Zhao, Wenjie Li, Xing Xie, and Minyi Guo.  
In Proceedings of The 2019 Web Conference (WWW 2019)

![](https://github.com/hwwang55/MKR/blob/master/framework.png)

MKR is a **M**ulti-task learning approach for **K**nowledge graph enhanced **R**ecommendation.
MKR consists of two parts: the recommender system (RS) module and the knowledge graph embedding (KGE) module. 
The two modules are bridged by *cross&compress* units, which can automatically learn high-order interactions of item and entity features and transfer knowledge between the two tasks.


### Files in the folder

- `data/`
  - `wine/`
    - `item_index2entity_id.txt`: the mapping from item indices in the raw rating file to entity IDs in the KG;
    - `kg.txt`: knowledge graph file;
    - `convert_rating.txt`: raw rating file of Last.FM;
- `src/`: implementations of MKR.




### Running the code
- Wine
  ```
  $ cd src
  $ python preprocess.py
  $ python main.py
  ```