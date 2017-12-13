# spectrum
**spectrum** is a library for knowledge fusion. Nowadays we have a lot of algorithms to extract structured
data from unstructured data sources such as text, table, even images. The extracted data
comes in RDF triples (subject, predicate, object). These triples are extracted from different
data sources, i.e., web sites and from different extractors. Naturally, some of them contain 
conflict facts since web sites can contain conflicting information.

**spectrum** allows you to develop your own algorithm and compare with other algorithms. 
Currently it implements the following algorithm:

* Majority Voting
* Truth Finder
* Accu 
* Hybrid
* LTM
* Spectrum 

## Fundmental concepts

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
   

## Datasets

We collect a number of datasets that were used in researches.

You can find the following datasets with their description in data/original/ together with their
descriptions.

1. **stock**
2. **books** 
3. **flights**
4. **restaurant**
5. **weather** 
   
These datasets are all collected from this [website](http://lunadong.com/fusionDataSets.htm). When you use
these datasets please cite their papers.

We also collect datasets from [here.](http://da.qcri.org/dafna/#/dafna/exp_sections/realworldDS/flight.html)
Most of the datasets found here overlaps with the above datasets. But they are already processed. 
You can find these processed versions in data/