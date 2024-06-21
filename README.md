# BCSC identification via RNA sequencing
Repository of the source code of my master degree thesis.

## Objects
- The training formats are stored a key-value pair dictionary with pickle. The key is the gene name while the value corresponds to the position of gene in the format.

- ENSG mapping: Transform ENSG gene name encoding to standard gene name, obtained base from ensembl.org/biomart and consequently cleaned and prepare for scripting.


## Source
- data: Utilities corresponding to loading / preprocessing the data
- experiments: Example script of one of the many training loops / scripts used.
- ml: Model declaration is defined here, along with custon dataset and some more utilities used in training.
- utils: Extra utilities, like both CytoTRACE and ORIGINS reimplementation with visualization (plots) scripts.
