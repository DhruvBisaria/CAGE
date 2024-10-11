# CAGE

The Cube-based Anomalous Gas Estimator (CAGE) is a code suite that uses KinMS to kinematically model the data cubes of galaxies and uses hierarchical clustering algorithms (Astrodendro) to identify and quantify gas in each frame that is anomalous with a thin disk approximation. I am currently working on making this suite publicly usable and am in the process of addressing some remaining bugs. 

Documentation to drop in the near future! 

In the meantime, you can get a sense of what this code does by taking a look at Chapter 3 of https://qspace.library.queensu.ca/items/7074b010-b3c1-4421-8710-c3eac5545c21

I have attached a folder with a sample galaxy data cube for any user to test its functionality. 
For now, simply run 'python3 CAGE_master.py sample_galaxy/sample_run_order.csv' to get started. The run_order csv contains all the other inputs necessary for the code to run.

CAGE runs on Python 3.7.4 has a GPL-3.0 License.

In the meantime, if you have any issues or questions, please PM me or raise an issue here on Github.

Best,

Dr. Dhruv Bisaria



