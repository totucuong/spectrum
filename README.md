# Spectrum

**spectrum** is a truth discovery library. It provide black-box truth discovery implementation.

Truth discovery refers to the process of reconciling conflicting data values that are provided by different data sources while also estimate the data source reliabilities.

Truth discovery algorithms falls into two categories. In the first category, iterative algorithms are used to infer correct values and source reliabilities. These iterative algorithms rely on one single key idea: if a value is provided by multiple reliable sources then there is a high chance it is correct. On the other hand if a source provides many correct values then it is considered as reliable. A typical workflow is that scores are initialized for source reliabilities as well as data value correctness. Then source reliabilities scores are used to update correctness scores of data values.

The second category of truth discovery algorithms uses statistical models. These models has two main advantages:

1. Its output has clear interpretation. Reliability and value scores are modeled as probability distributions. These enables downstream decision making.
2. Easy incorporation of domain knowledge via priors.

The statistical discovery problem has two disadvantages:

1. Scalability: a lot of them relies on MCMC algorithms which can be slow.
2. Model complex: it is not easy to specify and then build inference algorithm for statistical models.

We introduce variational inference black-box variational inference to solve the scalability problem and enable quick model experiments. VI cases inference problem into optimization problem which enables us to bring in stochastic gradient descent algorithm that scale with dataset size.



# Data Models
**Claim** a claim is a triple (source, object, value), where:

   1. *source* can be a website or a person that asserts a value about an object
   2. *object* can be a person's birthday, or the height of Mount Everest.
   3. *value* is the value of an object. For example `16/04/2001` is a birthday of someone.
   

# Datasets

We collect a number of datasets that were used in researches. They can be found in data/original/ together with their
descriptions.

1. **stock**
2. **books** 
3. **flights**
4. **restaurant**
5. **weather** 
6. **population**
   
These datasets are all collected from this [website](http://lunadong.com/fusionDataSets.htm). When you use
these datasets please cite their papers. We also collect datasets from [here.](http://da.qcri.org/dafna/#/dafna/exp_sections/realworldDS/flight.html).


# Install

If you want to use ``spectrum`` then you can create a conda environment using:

```
conda env create -f env.yml
```
This will create a conda enivornment called ``spectrum``

If you want to run ipython notebooks in the directory ``notebooks``, you need to install seaborn using the following command

```
conda install jupyterlab
conda install seaborn
```