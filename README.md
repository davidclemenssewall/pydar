# Processing repeat terrestrial laser scans with pydar

This package contains a set of classes and methods for classifying, filtering, and aligning repeat Terrestrial Laser Scanning (TLS) data, converting pointclouds to irregular or gridded surfaces, and visualizing and interacting with these data. The code was designed for sea ice with an ice fixed, lagrangian reference frame. However, it may be applied to terrestrial data as well and future updates may add georeferencing capabilities. The code has exclusively been developed and tested for a Riegl VZ1000 scanner although it should be adaptable to other terrestrial scanners.

Below is a non-exhaustive guide to processing and examining repeat TLS data using pydar

Terminology:
+ singlescan - TLS data from a single scan position (individual tripod placement)
+ project - All of the singlescans collected at the same field site on the same day (or 2 days in some cases). This is identical to RiSCAN's definition of a project.
+ scanarea - Multiple projects collected at the same field site on different days
+ project directory - The directory RiSCAN creates a project in (e.g. 'mosaic_rov_040120.RiSCAN')
+ SOP - Scanner's Own Position. 4x4 matrix that expresses the rigid transformation from the SOCS to the PRCS in homologous coordinates.
+ SOCS - Scanners Own Coordinate System. Coordinate system with the scanner as the origin.
+ PRCS - Project Coordinate System.

## Part 0: Prepare and export data from RiSCAN
0. Label reflectors. In the TPL (PRCS) rename tiepoints such that the reflectors have consistent names from project to project. I.e. 'r01' refers to the same physical point in space in all projects at a particular scanarea.
1. Export all scans as LAS 1.4 files. Create a directory within the project directory named 'lasfiles'. In RiSCAN, right click on a scan and click 'Export'. In the popup menu, click back and select (ctrl+click) all of the single scans. Make sure combine data is not checked. Scans should be exported in the Scanners Own Coordinate System (SOCS). Browse to the 'lasfiles' directory that you created and export scans. A las file should be generated for each singlescan.
2. Export project tiepoints. Click 'Export tiepoints' option in the TPL (PRCS). Name the file 'tiepoints.csv' and place it in the project directory.
3. Export SOP matrices. Under 'Registration' tab, go to 'Bulk SOP export'. Export all SOP matrices as '.DAT' files in the project directory.

## Part 1: Loading and displaying a project in pydar
The following script shows a basic example of how to read in a project and display it in an interactive window.

## Part 2: Filtering wind-blown snow particles
These tools are components of the ![FlakeOut](https://doi.org/10.5281/zenodo.5657286) filter. For more details see Clemens-Sewall et al 2022 (TODO: replace preprint with publication when accepted...)

0. The following script demonstrates applying a simple visibility based filter to label points that are disconnected from other points (i.e. floating in space).