# Spectrum
**spectrum** is a truth discovery library. It provide black-box truth discovery implementation.

### Data Models

**Claim** a claim is a triple (subject,predicate,object,confidence,source), where

   1. *subject* is a named entity such as "Obama", or "Hanoi".
   2. *predicate* is a relation between *subject* and *object*
   3. *object* is the value of (subject,predicate). For example, a claim (Obama, bornIn, USA)
   have "Obama" as the subject, "bornIn" as the predicate, "USA" as the object. In practice,
   object is not named entity per se, but it can extends to be a numerical value, or a date.
   4. *confidence* is a numerical score that express the belief degree in the claim. It can be
   a probability or just a nonpositive values. Typically
   this score comes from an extractor. If we do not have the confidence, or just want to ignore
   them then we can set them all to be 1.
   5. *source* is an id that represents the data sources that provide the claim. It could
   be just *s1* or *extractor1_url1*, which means this claim was produced by *extractor1* from
   web site at *url1*.
   

### Datasets

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