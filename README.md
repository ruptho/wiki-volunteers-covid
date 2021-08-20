# Volunteer contributions to Wikipedia increased during COVID-19 mobility restrictions
Repository for the preprint "Volunteer contributions to Wikipedia increased during COVID-19 mobility restrictions" (https://arxiv.org/abs/2102.10090).

This repository contains Python Code for experiments conducted during the creation of the paper, as well code for dataset retrieval and preprocessing. Under data/results we furthermore provide the final product of the preprocessing pipeline (aggregated.tsv.bz2), on which we perform the bulk of our experiments.

## Code, Analysis, and Data
The following notebooks demonstrate the results of the study, while importing most of their functionality from the python-scripts under the "helpers"-Folder:
- 0_download.ipynb: automatically downloads mediawiki history dumps of the studied Wikipedias
- 1_generate_data.ipynb: preprocesses data and saves intermediary files (e.g., aggregated.tsv.bz2)
- 2_did: difference-in-differences analysis and visualization of the results
- 2_vis: other visualizations, such as the intro-figure of the paper and ratio of COVID-19-related articles

Helper code files are structured as follows:
- logger.py: logging functionality
- preprocessing.py: contains the preprocessing pipeline
- retrieval.py: Interface for the active editor data in the Wikimedia REST API
- plot.py: visualization and plotting functionality
- analysis.py: code for producing difference-in-differences regression and other computations
- vars.py: helper variables, during reproduction of results (or preprocessing) please change the paths in here to your desired input and output directories 
- files.py: util for reading and writing files

The following folders contain additional information:
- data: input and output files of the preprocessing pipeline (generated paths depend on the set directories in vars.py)
- csv/did: did statistics (coefficients, std.err.) and results
- img: plots generated during the analysis and ouput of results
- data/results: contains preprocessed data (aggregated.tsv.bz2)

## Demo for processing arbitrary Wikipedia language editions
You can find a demo of how one would analyze any arbitrary language editions using our methodology under the "demo"-Folder.

In the specific use case depicted there, we compute Wikipedia language editions covering large parts of Eastern Europe (Polish, Czech, Ukrainian, and Russian Wikipedia). The data is first downloaded, then preprocessed, and lastly analyzed before results are being visualized in the notebooks and saved to the given subfolders (did/demo/, img/demo/).

The notebooks in this folder repurpose the code and analysis from the main notebooks. This demo also demonstrates which parts of the code can be modified to customize, for example, details of the analysis and visualization.
