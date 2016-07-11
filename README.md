#Generative Moment Matching Networks (GMMNs)
This is the code we used for the following paper:
* Yujia Li, Kevin Swersky, Richard Zemel.  *Generative moment matching networks*.  In International Conference on Machine Learning (ICML), 2015.

If you use this code in your research you should cite the above paper.

### Dependencies
To use the code you need to install some dependencies first:
* Standard python packages like **numpy, scipy, matplotlib**.  matplotlib is only needed for visualization.  You may also need sklearn for some features.
* [**gnumpy**](http://www.cs.toronto.edu/~tijmen/gnumpy.html).  If you have a NVIDIA GPU gnumpy can speed up your computation significantly.  To use GPUs you need to install [**cudamat**](https://github.com/cudamat/cudamat) first.  If you don't have a GPU you can use [**npmat**](http://www.cs.toronto.edu/~ilya/npmat.py) as a replacement for cudamat, then all computations will be done on a CPU.
* The authors' lightweight neural network and optimization packages [**pynn**](https://github.com/yujiali/pynn) and [**pyopt**](https://github.com/yujiali/pyopt).

Once you get all dependencies ready, try to run `python test.py`.  If you are running this with npmat then all tests should pass.  If you are running this on a GPU with cudamat then some tests will fail - this is expected because of the low numeric precision supported by cudamat (`float32` every where), but all tests should run and finish properly.

### Prepare data
Prepare the MNIST and TFD data, then go into the `dataio` directory, change paths to the datasets in `mnist.py` and `tfd.py`.

### Train the models
Use `python train.py -m <mode>` to train the corresponding model.  `<mode>` can be `mnistinput`, `mnistcode`, `tfdinput`, `tfdcode`, corresponding to the input space model and autoencoder code space model for the two datasets.

##### Other resources
There is a tensorflow implementation of GMMN provided by Siddharth Agrawal: https://github.com/siddharth-agrawal/Generative-Moment-Matching-Networks
