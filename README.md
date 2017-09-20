# Adapting Object Detectors 

Implementation of the paper [Adapting Object Detectors from Images toWeakly Labeled Videos](http://www.cs.umanitoba.ca/~ywang/papers/bmvc17_adapt.pdf). 

### Usage
The first step of our approach is to generate a shortlist of object proposals from source and target images. We use the edge boxes algorithm for generating the object proposals. 
Let K be the number of object proposals generated on the image. We represent each proposal as a 4096-dimensional CNN feature vector. 

### Run the experiment
1. Get into each folder named as class name [i.e. 01_aeroplane].
2. Keep the .t7 file of the dataset with data, ground truth bounding box, class label and proposals.
3. Run doall.lua file using th command [i.e. th ../doall.lua].
4. After training, run evaluate.sh file for getting the corloc for each class.

### Model Architecture

![model architecture](./net.png)

### Citation

If you find this project useful for your work, please consider cite the paper.
```
@article{BMVC2017Adapt,
  author = {Omit Chanda, Eu Wern Teh, Mrigank Rochan, Zhenyu Guo and Yang Wang},
  title = {Adapting Object Detectors from Images to Weakly Labeled Videos},
  journal = {The 28th British Machine Vision Conference (BMVC), 2017},
  year = {2017}
}
```



