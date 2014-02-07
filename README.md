CMOS-python-tutorial
====================

Python tutorial at [CMOS 2014](http://www.cmos.ca/congress2014/index.php/en/)


### To launch the presentation locally

    ./present.sh

### To author the presentation in the IPython notebook    

In the project folder launch ipython:

    ipython notebook --pylab inline
    
The page with a list of notebooks will appear in your browser window. Open the notebook `cmos2014-python-tutorial.ipynb` and switch to slideshow view to see how the content is broken into slides.


###Dependencies

* IPython
* matplotlib
* basemap
* GDAL
* pillow
* fiona
* netcdf4-python
* scipy
* numpy
* numexpr
* cython
* pandas
* Maybe iris and cartopy if decide to cover..


The list above are the python packages and some of them have C/Fortran dependencies:

* netcdf4-python : NetCDF4 and HDF5 (can be installed using `apt-get install ...`)
* scipy : requires fortran compiler gfortran 
* basemap: requires geos (install libgeos-dev using apt-get)
* GDAL and Fiona: require libgdal (install libgdal-dev using apt-get )

