# GradCAM visualizations

### Setup

You may need a few more packages to be installed in the environment used for the main experiments:

```bash
pip install matplotlib
pip install opencv-python
pip install torchray
```

### Generate the GradCAMs and the superimposed images

You can generate the visualizations for different descriptions as in Fig. 1, 4, 5 and 6 from the paper by running the gradcams.ipynb notebook. The results will be stored in a folder called "results".

### Analyze the results

In this part you can check the ratio between the usage of core and spurious features in prediction. 

Firstly, make sure the "results" folder is in the same directory as "analyze.ipynb". In case you are using colab you can upload the "results.zip" file there.

Now, by running the "analyze.iypnb" notebook you can create the segmentation masks and analyze the percentage of core and spurious regions used in classification.

