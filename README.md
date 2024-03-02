# VGNet: Multimodal Feature Extraction and Fusion Network for 3D CAD Model Retrieval
## Authors: Feiwei Qin, Gaoyang Zhan, Meie Fang, C.L.Philip Chen, Ping Li


![](https://img.shields.io/github/contributors/divanoLetto/3D_STEP_Classification?color=light%20green) ![](https://img.shields.io/github/repo-size/divanoLetto/3D_STEP_Classification)

## Abstract
*The reuse of 3D CAD models is crucial for industrial manufacturing as it shortens development cycles and reduces costs. Deep learning based 3D model retrieval has made significant progress. There are many representations for 3D models, among which multi-view representation has demonstrated superior retrieval performance. However, directly applying these 3D model retrieval approaches to 3D CAD model retrieval may result in issues, such as losing engineering semantic and structural information. In this paper, we find that multi-view and B-rep can complement each other and therefore propose the VGNet (View Graph neural Network), which effectively combines multi-view and B-rep to accomplish 3D CAD model retrieval. More specifically, based on the characteristics of the regular shape of 3D CAD models and the richness of attribute information in the B-rep attribute graph, we design two feature extraction networks for each modality separately. Meanwhile, to explore the latent relationship between multi-view and B-rep attribute graph, the multi-head attention enhance module is designed. Furthermore, the multimodal fusion module is adopted to make the joint representation of 3D CAD models more discriminative by using a correlation loss function. Experiments are carried out on the real manufacturing 3D CAD dataset and public dataset to validate the effectiveness of the proposed approach.*

Details about the implementation and the obtained results can be found in the `docs` folder.

---

## Installation

1. Create Conda virtual environment:

    ```
    conda create --name 3D_STEP_Classification python=3.8
    conda activate 3D_STEP_Classification
    ```
    
2. Clone this repository:
    ```
    git clone https://github.com/divanoLetto/3D_STEP_Classification
    ```
3. Install CUDA Toolkit version 11.3 from the [official site](https://developer.nvidia.com/cuda-11.3.0-download-archive).

4.  Install the following requirements:
    ```
    conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
    conda install pyg -c pyg
    conda install -c conda-forge tensorboardx
    conda install -c anaconda scikit-learn
    conda install -c conda-forge matplotlib
    conda install -c anaconda scikit-image
    conda install -c conda-forge pythonocc-core
    ```

5. Finally, make sure to obtain the [Traceparts STEP dataset](https://drive.google.com/drive/folders/1jV1B5Y8XmGY-XhjildX2BdYTEFtLK5XQ?usp=sharing), extract the STEP models and save them in the `/Datasets/` folder.

# Usage

The program implements the classification and retrieval of 3D models through an approach based on graphs obtained from STEP files and the [MVCNN](https://github.com/jongchyisu/mvcnn_pytorch) approach based on multiple 2D views.

## Graph classification and retrieval

For the graph based approach, to convert a 3D STEP dataset into a Graph dataset, run the script:    
```
$ python step_2_graph.py
```    
It takes two arguments: `--path_stp` specifies the path of the input STEP dataset and `--path_graph` specifies the output path where the graph dataset will be saved.
Then for the classification task on the relised dataset run the script:   
```
$ python train_GCN.py
```
It takes 5 arguments: `--run_folder` indicates the run directory, `--learning_rate` sets the strating learning rate, `--batch_size` sets the batch size, `--num_epochs` sets the number of traing epochs, `--dropout` the dropout probability.    
Alternatively, we provide the `Graph_classification.ipynb` ipython notebook, that performs both the dataset conversion and graph classification task.   
A Graph Convolutional Neural Network model trained for the classification task in this way can then be used for the retrieval task by running the `Graph_retrieval.ipynb` script.

## Multi-views classification 

For the multi 2D views  based approach, to convert each 3D model into a 12 2D views,  run the script:
```
$ python step_2_multiview.py 
```
It takes two arguments: `--path_stp` specifies the path of the input STEP dataset and `--path_multiview` specifies the output path where the multi-views dataset will be saved.   
Then for the classification task run the script:
```
$ python train_mvcnn.py
```
It takes 10 arguments: `--num_models` indicates the number of models per class, `--lr` sets the strating learning rate, `--bs` sets the batch size, `--weight_decay` sets the weight decay ratio of the learning rate, `--num_epoch` sets the number of training epochs, `--no_pretraining` indicates if the base net will start pretrained or not, `--cnn_name` the net name, num_views the number of 2D views, `--train_path` specifies the path of the train data, `--test_path` specifies the path of the test data, `--val_path` specifies the path of the validation data.   
Alternatively, we provide a the `MultiViews_Classification.ipynb.ipynb` ipython notebook, that performs both the dataset conversion and multi-views classification task. 
Similarly to the graph-based approach, a model trained for classification task can then be used for the 3D retrieval task.

---

# Repository Requirements

This code was written in Pytorch 1.11. with CUDA Toolkit version 11.3 to enable GPU computations. We recommend setting up a virtual environment using [Miniconda](https://docs.conda.io/en/latest/miniconda.html). Python 3.8 is required for the PythonOCC library needed for the conversion from STEP to the multi-views data.

## Data Organization

The following is the organization of the dataset directories expected by the code:

* data **root_dir**/
  * **dataset** name/ (eg Traceparts)
    * STEP_models (all of the 3D STEP models divided by class)
      * Class 0 (all STEP models of the class 0)
      * Class 1 (all STEP models of the class q)
      * ...
    * graphml_models (all of the converted graphml models divided by class)
      * Class 0 (all graphml models of the class 0)
      * Class 1 (all graphml models of the class 1)
      * ... 
    * MVCNN_models (all of the converted multi-views 2D images divided by class)
      * Class 0
        * train (the train set 2D views of the class 0)
        * test (the test set 2D views of the class 0)
        * valid (the validation set 2D views of the class 0)
      * ...

# Cite

Please consider citing our work if you find it useful:

```
@misc{https://doi.org/10.48550/arxiv.2210.16815,
  doi = {10.48550/ARXIV.2210.16815},
  url = {https://arxiv.org/abs/2210.16815},
  author = {Mandelli, L. and Berretti, S.},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {CAD 3D Model classification by Graph Neural Networks: A new approach based on STEP format},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```
