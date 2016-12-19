# Traffic-Viewer
Open source solution for gathering statistics on road traffic (cameras first) and processing them.

# Goal
The first step of the project will focus on extracting basic information from image sequences recorded by traffic cameras :
  - Number 
  - Position
  - Average celerity
  - Individual speed (if acurate)

Accuracies of those statistics and computing time will be the main focus of this project, the long term goal being running several instances of the program on the same device, to gather and process flows from different sources.

# Methodology
The first part of that work will be implemented in python2 with opencv to easily investigate different methods, and then will be implemented in C++ to ensure good real time performances. 
