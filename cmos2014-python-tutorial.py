# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

#Necessary imports
import os
from netCDF4 import Dataset

# <markdowncell>

# Data handling and visualization using Python
# =============================================
# 
# 
# <span id="authors">
# Oleksandr (Sasha) Huziy, Johnathan Doyle, Martin Deshaies-Jacques
# <span>
# 
# CMOS, June 2014
# 
# <table class="logo">
#     <tr><td>
#         <img src="files/images/logo_2.png"/> <img src="files/images/crsng.png"/> <img src="files/images/logo_uqam_0.png"/>
#     </td></tr>
# </table>

# <markdowncell>

# Outline part I
# ========
# 
# * **Python basics**
# 
#     * Builtin data types
#     
#     * Operations on file system, strings and dates
#     
#     * Modules, classes and functions
# 
# 
# * **Libraries for scientific computing**
# 
#     * NumPy/SciPy
# 
# 
# 
# * **Handling NetCDF4 files**
#     
#     * Netcdf4-python

# <markdowncell>

# Introduction and history
# =====
# 
# ### Python is an interpreted, strictly typed programming language developed by *Guido Van Rossum*.
# 
# 
# ### It was created in 90s of the previous century and has seen 3 major releases from that time. 
# 
# ### There many implementations of Python (in C, Java, Python, ...), CPython implementation is used most widely and we are going to use it during the tutorial.
# 
# ### Talk on Python history by Guido Van Rossum [here](http://www.youtube.com/watch?v=ugqu10JV7dk).
# 
# 
# 

# <markdowncell>

# Syntax 
# =======

# <codecell>

#Variable decalaration <-This is a comment
a = 10; b = "mama"; x = None;

x = [1,2,4, 5, "I am a fifth element"]
#looping
for el in x:
    #Checking conditions
    if isinstance(el, int):
        print(el ** 2 % 10),
    else:
        msg = ", but my index is {0}."
        msg = msg.format(x.index(el))
        print(el + msg),

#now we are outside of the loop since no identation
print "\nSomething ..."

# <markdowncell>

# Syntax 
# =======

# <codecell>

#accessing elements of a list
x[3], x[-1], x[1:-1]

# <markdowncell>

# ##Defining functions

# <codecell>

def myfirst_func(x, name = "Sasha"):
    """
    This is the comment describing method 
    arguments and what it actually does
    x - dummy argument that demonstrates 
    use of positional arguments
    name - demonstrates use of keyword arguments
    """
    print "Hello {0} !!!".format(name)
    return 2, None, x

# <markdowncell>

# ##Calling functions

# <codecell>

#Calling the function f
myfirst_func("anything", name = "Oleks") 

# <markdowncell>

# Python basics - code hierarchy
# ==============
# 
# + Each python file is a python module, it can contain function and class definitions
# 
# 
# + A folder with python files and a file ```__init__.py``` (might be empty file) is called package
# 

# <markdowncell>

# Exercises on basic syntax
# =====
# 
# * Create a script `hello.py` in your favourite editor and add `print "Hello world"`, save and run it: `python hello.py`
# 
# 
# * Modify the script making it print squares of odd numbers from 13 to 61 inclusively.
# 
# 
# * What will this code print to console:
# 
# 
# <pre class="co">
#     x = 1 #define a global variable
#     def my_function(argument = x):
#         print argument ** 2
#     #change the value of the global variable
#     x = 5
#     my_function() #call the function defined above
# </pre>

# <codecell>

x = 1 #define a global variable
def my_function(argument = x):
    print argument ** 2
#change the value of the global variable
x = 5
my_function() #call the function defined above

# <markdowncell>

# Builtin data containers
# =============
# ##Python provides the following data structures. Which are actually classes with attributes and methods.
# 
# * `list` (range, *, +, pop, len, accessing list elements, slices, last element, 5 last elements)
# 
#   
# * `tuple` (not mutable, methods)
# 
# 
# * `dict` (accessing elements, keys, size)
# 
# 
# * `set` (set theoretical operations, cannot have 2 equal elements)
# 

# <headingcell level=1>

# Builtin data containers (lists)

# <codecell>

#lists (you can conactenate and sort in place)
the_list = [1,2,3,4,5]; other_list = 5 * [6];
print the_list + other_list

# <codecell>

#Test if a number is inside a list
print 19 in the_list, 5 in the_list, (6 in the_list and 6 in other_list)

# <headingcell level=1>

# Builtin data containers (lists)

# <codecell>

#square eleaments of a list
#list comprehension
print [the_el ** 2 for the_el in the_list] 

# <codecell>

#Generating list or iterable of integers
print range(1,20)

# <headingcell level=1>

# Builtin data containers (lists)

# <codecell>

##There are some utility functions that can be applied to lists
print sum(the_list), \
      reduce(lambda x, y: x + y, the_list)

# <codecell>

#loop through several lists at the same time
for el1, el2 in zip(the_list, other_list):
    print(el1+el2),

# <markdowncell>

# Builtin data containers (tuples)
# ======

# <codecell>

the_tuple = (1,2,3) #tuple is an immutable list, is hashable
print the_tuple[-1]

# <markdowncell>

# Builtin data containers (tuples)
# ======

# <codecell>

#tuples are immutable, e.g:
try:
    the_tuple[1] = 25
except TypeError, te:
    print te

# <markdowncell>

# Builtin data containers (dictionary)
# =====

# <codecell>

#dictionary
author_to_books = { 
"Stephen King": 
    ["Carrie","On writing","Green Mile"],

"Richard Feynman": 
    ["Lectures on computation", 
     "The pleasure of finding things out"]
}
#add elements to a dictionary
author_to_books["Andrey Kolmogorov"] = \
        ["Foundations Of The Theory Of Prob..."]

# <markdowncell>

# Builtin data containers (dictionary)
# =====

# <codecell>

#print the list of authors
print author_to_books.keys()

# <codecell>

#Iterate over keys and values
for author, book_list in author_to_books.iteritems():
    suffix = "s" if len(book_list) > 1 else ""
    print("{0} book{1} by {2};\n".format(len(book_list), suffix, author)),

# <markdowncell>

# Exercises: builtin containers
# ======

# <markdowncell>

# * Find a sum of squares of all odd integers that are smaller than 100 in one line of code (Hint: use list comprehensions)
# 
# 
# * Find out what does `enumerate` do.
# 
# 
# * Implement recursive fibonacci function with caching, which given the index of a fibonacci number returns its value.

# <markdowncell>

# Modules and classes to operate on
# ==============
# 
# 
# * File system (```os, shutil, sys```)
# 
#     * create folder, list folder contents, check if file or folder exists 
# 
# 
# * Strings (+, *, join, split, regular expressions) 
# 
# 
# * Dates (datetime, timedelta, ) 

# <markdowncell>

# #File system

# <codecell>

import os
#print current directory
print os.getcwd()

# <markdowncell>

# #File system

# <codecell>

#get list of files in the current directory
flist = os.listdir(".")
print flist[:7]

# <markdowncell>

# #File system

# <codecell>

#Check if file exists
fname = flist[0]
print fname,":", os.path.isfile(fname), \
      os.path.isdir(fname),  \
      os.path.islink(fname)

# <markdowncell>

# You might also find useful the following modules: `sys, shutil, path` 

# <markdowncell>

# #Strings

# <codecell>

s = "mama"
#reverse (also works for lists)
print s[-1::-1]

# <codecell>

#Dynamically changing parts of a string
tpl = "My name is {0}. I am doing my {1}.\nI am {2} old.\nWeight is {3:.3f} kg"
print tpl.format("Black", "PhD", 25, 80.7823)

# <markdowncell>

# #Strings

# <codecell>

#Splitting
s = "This,is,a,sentence"
s_list = s.split(",")
print s_list

# <codecell>

#joining a list of strings
list_of_fruits = ["apple", "banana", "cherry"]
print "I would like to eat {0}.".format(" or ".join(list_of_fruits))

# <markdowncell>

# #Strings: regular expressions

# <codecell>

#regular expressions module
import re
msg = "Find 192, numbers 278: and -7 and do smth w 89"
groups = re.findall(r"-?\d+", msg)
print groups 
print [float(el) for el in groups] #convert strings to floats

# <codecell>

#regular expressions module
groups = re.findall(r"-?\d+/\d+|-?\d+\.\d+|-?\.?\d+|-?\d+\.?", "Find 192.28940, -2/3 numbers 278: and -7 and .005 w 89,fh.5 -354.")
print groups

# <markdowncell>

# # Dates

# <codecell>

#What time is it
from datetime import datetime, timedelta
d = datetime.now(); print d

# <codecell>

#hours and minutes
d.strftime("%H:%M"), d.strftime("%Hh%Mmin")

# <markdowncell>

# # Dates: How long is the workshop?

# <codecell>

start_date = datetime(2013, 12, 17, 9)
end_date = datetime(2013, 12, 18, 17)
print end_date - start_date

# <codecell>

#you can mutiply the interval by an integer
print (end_date - start_date) * 2

# <markdowncell>

# Exercises: Operations with strings, dates and file system
# ======
# 
# * Get list of all items in the folders from your `LD_LIBRARY_PATH` environment variable
# 
# 
# * Write a function that finds all the numbers in a string using `re` module (Note: make sure real numbers are also accounted for. Optionally you can also account for the fractions like 2/3)
# 
# 
# * Calculate your age in weeks using `timedelta`
# 
# 
# * Figure out on which day of week you were born (see the `calendar` module), you can have fun by determining on which day of week you'll have your birthday in 2014.

# <markdowncell>

# NumPy
# ========
# The library for fast manipulations with big arrays that fit into memory.

# <codecell>

import numpy as np
#Creating a numpy array from list
np_arr = np.asarray([1,23,4, 3.5,6,7,86, 18.9])
print np_arr

# <markdowncell>

# NumPy
# ========

# <codecell>

#Reshape
np_arr.shape = (2,4)
print np_arr

# <codecell>

#sum along a specified dimension, here 
print np_arr.sum(axis = 1)

# <markdowncell>

# Numpy
# =====

# <codecell>

#create prefilled arrays
the_zeros = np.zeros((3,9))
the_ones = np.ones((3,9))

print the_zeros
print 20 * "-"
print the_ones

# <markdowncell>

# Numpy provides many vectorized functions to efficiently operate on arrays
# =====

# <codecell>

print np.sin(np_arr)

# <codecell>

print np.cross([1,0,0], [0,1,0])

# <markdowncell>

# Numpy: fancy indexing
# =====

# <codecell>

arr = np.random.randn(3,5)
print "Sum of positive numbers: ", arr[arr > 0].sum()

print "Sum over (-0.1 <= arr <= 0.1): ", \
       arr[(arr >= -0.1) & (arr <= 0.1)].sum()

print "Sum over (-0.1 > arr) or (arr > 0.1): ", \
       arr[(arr < -0.1) | (arr > 0.1)].sum()

# <markdowncell>

# Exercises: Numpy
# =====
# 
# * Generate a 10 x 3 array of random numbers (in range \[0,1\]). For each row, pick the number closest to 0.5 ([source](http://scipy-lectures.github.io/intro/numpy/exercises.html#crude-integral-approximations)) *Hint:* use `np.argmin`.
# 
# 
# * Checkout numpy for MATLAB users transition [table](http://wiki.scipy.org/NumPy_for_Matlab_Users).

# <codecell>

arr = np.random.randn(10,3)
d = np.abs(arr - 0.5)

print arr
colinds = np.argmin(d, axis = 1)
rowinds = range(arr.shape[0])
arr[rowinds, colinds]

# <markdowncell>

# SciPy
# =======
# 
# ## [Scipy lectures](http://scipy-lectures.github.io/)

# <markdowncell>

# SciPy packages
# =======
# 
# It contains many paackages useful for data analysis
# 
# |Package| Purpose|
# |:-------:|:----------:|
# | scipy.cluster   | Vector quantization / Kmeans |
# | scipy.constants |  Physical and mathematical constants | 
# | scipy.fftpack   | Fourier transform | 
# | scipy.integrate|  Integration routines | 
# | scipy.interpolate|  Interpolation | 
# | scipy.io | Data input and output | 
# | scipy.linalg|  Linear algebra routines | 
# | scipy.ndimage|  n-dimensional image package | 

# <markdowncell>

# SciPy packages (cont.)
# =======
# 
# It contains many paackages useful for data analysis
# 
# |Package| Purpose|
# |:-------:|:----------:|
# | scipy.odr |  Orthogonal distance regression | 
# | scipy.optimize |  Optimization | 
# | scipy.signal |  Signal processing | 
# | scipy.sparse |  Sparse matrices | 
# | scipy.spatial |  Spatial data structures and algorithms | 
# | scipy.special |  Any special mathematical functions | 
# | scipy.stats |  Statistics | 

# <markdowncell>

# SciPy: ttest example
# =======

# <codecell>

from scipy import stats
#generating random variable samples for different distributions
x1 = stats.norm.rvs(loc = 50, scale = 20, size=100)
x2 = stats.norm.rvs(loc=5, scale = 20, size=100)
print stats.ttest_ind(x1, x2)

x1 = stats.norm.rvs(loc = 50, scale = 20, size=100)
x2 = stats.norm.rvs(loc=45, scale = 20, size=100)
print stats.ttest_ind(x1, x2)

# <codecell>

font_size = 20
params = {
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
}
plt.rcParams.update(params) #set back font size to the default value

# <markdowncell>

# SciPy: integrate a function
# =====
# $$
# \int\limits_{-\infty}^{+\infty} e^{-x^2}dx \;=\; ?
# $$

# <codecell>

from scipy import integrate

# <codecell>

def the_func(x):
    return np.exp(-x ** 2)
print integrate.quad(the_func, -np.Inf, np.inf)

# <codecell>

print "Exact Value sqrt(pi) = {0} ...".format(np.pi ** 0.5)

# <markdowncell>

# Exercises: SciPy
# =======
# 
# * Calculate the integral over the cube D=\[0,1\] x \[0, 1\] x \[0,1\] ([source](http://scipy-lectures.github.io/intro/numpy/exercises.html)):
# 
#     $$
#     \int\int\limits_{D}\int \left(x^y -z \right) dD
#     $$
# 
#     *Hint: checkout* `scipy.integrate.tplquad`.

# <codecell>

def the_func_to_integr(x,y,z):
    return x**y - z

integrate.tplquad(the_func_to_integr, 0,1, lambda x: 0, lambda x: 1, lambda x,y: 0, lambda x,y: 1)

# <markdowncell>

# Netcdf4-python
# =================
# ##The python module that for reading and writing NetCDF files in python, created by *Jeff Whitaker*. 
# ##Requires installation of C libraries:
# 
# * **NetCDF4**
# 
# 
# * **HDF5**
# 
# 

# <markdowncell>

# #Netcd4-python
# 
# ## Below is the example of creating a netcdf file using the **netcdf4-python** library.

# <codecell>

from netCDF4 import Dataset
file_name = "test.nc"
if os.path.isfile(file_name):
    os.remove(file_name)
    
#open the file for writing, you can Also specify format="NETCDF4_CLASSIC" or "NETCDF3_CLASSIC"
#The format is NETCDF4 by default
ds = Dataset(file_name, mode="w")

# <markdowncell>

# Netcdf4-python: create dimensions
# ==========

# <codecell>

ds.createDimension("x", 20)
ds.createDimension("y", 20)
ds.createDimension("time", None)

# <markdowncell>

# Netcdf4-python: create variables
# ==========

# <codecell>

var1 = ds.createVariable("field1", "f4", ("time", "x", "y"))
var2 = ds.createVariable("field2", "f4", ("time", "x", "y"))

# <markdowncell>

# #Write actual data to the file

# <codecell>

#generate random data and tell to the program where it should go
data = np.random.randn(10, 20, 20)
var1[:] = data
var2[:] = 10 * data + 10
#actually write data to the disk
ds.close();

# <markdowncell>

# Netcdf4-python: reading a netcdf file
# ==============
# 
# Open the netcdf file for reading:

# <codecell>

from netCDF4 import Dataset
ds = Dataset("test.nc")

# <markdowncell>

# Select variables of interest: no data loading happens at this point
# ======

# <codecell>

#what variables are in the file
print ds.variables.keys()

#now data is a netcdf4 Variable object, which contain only links to the data
data1_var = ds.variables["field1"]
data2_var = ds.variables["field2"]
#You can query dimensions and shapes of the variables
print data1_var.dimensions, data1_var.shape

# <markdowncell>

# Read data from the netcdf file for corresponding variables and time steps
# ==========

# <codecell>

#now we ask to really read the data into the memory
all_data = data1_var[:]
#print all_data.shape
data1 = data1_var[1,:,:]
data2 = data2_var[2,:,:]
print data1.shape, all_data.shape, all_data.mean(axis = 0).mean(axis = 0).mean(axis = 0)

# <markdowncell>

# Outline part II
# ========
# 
# * **Plotting libraries**
# 
#     * Matplotlib
# 
#     * Basemap
#     
#     
# * **Grouping and subsetting temporal data with ```pandas```**    
# 
# 
# * **Interpolation using ```cKDTree``` (```KDTree```) class**
# 
# 
# * **Speeding up your code**

# <markdowncell>

# Matplotlib
# ============
# 
# ##The module for creating publication quality plots (mainly 2D), created by *John Hunter*.
# 
# ##[Matplotlib gallery](http://matplotlib.org/gallery.html)
# 
# 
# An alternative is PyNGL - a wrapper around NCL developed at NCAR.
# 

# <markdowncell>

# Matplotlib
# =====
# Example taken from the matplotlib library and modified.
# 
# Read some timeseries into memory and import external dependencies:

# <codecell>

import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.cbook as cbook
from matplotlib.dates import strpdate2num
from matplotlib.dates import DateFormatter
from matplotlib.dates import DayLocator, MonthLocator

datafile = cbook.get_sample_data('msft.csv', asfileobj=False)
dates, closes = np.loadtxt(datafile, delimiter=',',
    converters={0: strpdate2num('%d-%b-%y')},
    skiprows=1, usecols=(0,2), unpack=True)

# <markdowncell>

# Plot timeseries
# ===================

# <codecell>

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(dates, closes, lw = 2);

# <markdowncell>

# Format the x-axis properly
# ====

# <codecell>

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(dates, closes, lw = 2)

ax.xaxis.set_major_formatter(DateFormatter("%b\n%Y"))
ax.xaxis.set_minor_locator(DayLocator())
ax.xaxis.set_major_locator(MonthLocator())

# <markdowncell>

# Modify the way the graph looks
# =====

# <codecell>

def modify_graph(ax):
    ax.xaxis.set_major_formatter(DateFormatter("%b"))
    ax.xaxis.set_minor_locator(DayLocator())
    ax.xaxis.set_major_locator(MonthLocator())
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid()
    ax.set_ylim([25.5, 30]); ax.set_xlim([dates[-1], dates[0]]);

# <markdowncell>

# Modify the way the graph looks
# =====

# <codecell>

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(dates, closes, "gray", lw = 3)
modify_graph(ax)

# <markdowncell>

# Let us draw the data we just saved to the netcdf file.
# ====
# 
# Do necessary imports and create configuration objects.

# <codecell>

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib import cm

levels = [-30,-10, -3,-1,0,1,3,10,30,40]
bn = BoundaryNorm(levels, len(levels) - 1)
cmap = cm.get_cmap("jet", len(levels) - 1);

# <markdowncell>

# Matplotlib: actual plotting
# ============

# <codecell>

font_size = 20
params = {
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'figure.figsize' : (10, 3)
}
plt.rcParams.update(params) #set back font size to the default value

def apply_some_formatting(axes, im):
    axes[1].set_yticks([]); 
    axes[2].set_aspect(20); 
    cax = axes[2]; 
    cax.set_anchor("W")
    cb = plt.colorbar(im2, cax = axes[2]);

# <codecell>

fig, axes = plt.subplots(nrows=1, ncols=3)
im1 = axes[0].contourf(data1.transpose(), 
                       levels = levels, 
                       norm = bn, cmap = cmap)
im2 = axes[1].contourf(data2.transpose(), 
                       levels = levels, 
                       norm = bn, cmap = cmap)
apply_some_formatting(axes, im2)

# <markdowncell>

# Execises: Matplotlib
# =============
# 
# * Reproduce the panel plot given in the example. (Try different colormaps)
# 
# 
# * Plot a timeseries of daily random data. (Show only month names as Jan, Feb, .. along the x-axis, make sure they do not overlap)

# <markdowncell>

# Pandas
# ====
# 
# * ## Initially designed to process and analyse long timeseries
# 
# 
# * ## Author: *Wes McKinney*
# 
# * ## Home page: [pandas.pydata.org](http://pandas.pydata.org/) 

# <markdowncell>

# Load the same timeseries as for matplotlib example
# ======

# <codecell>

import pandas as pd
datafile = cbook.get_sample_data('msft.csv', asfileobj=False)
df = pd.DataFrame.from_csv(datafile); df.head(3)

# <markdowncell>

# Select the closes column
# ====== 

# <codecell>

closes_col_sorted = df.Close.sort_index()
closes_col_sorted.plot(lw = 2);

# <markdowncell>

# Group by month and plot monthly means
# ====== 

# <codecell>

df_month_mean = df.groupby(by = lambda d: d.month).mean()
df_month_mean = df_month_mean.drop("Volume", axis = 1)
df_month_mean.plot(lw = 3);
plt.legend(loc = 2);

# <markdowncell>

# It is easy to resample and do a rolling mean
# ====== 

# <codecell>

resampled_closes = closes_col_sorted.resample("5D", how=np.mean)
ax = pd.rolling_mean(resampled_closes, 4)[3:].plot(lw = 2);

# <markdowncell>

# Customizing the graph produced by pandas
# ====== 

# <codecell>

ax = pd.rolling_mean(resampled_closes, 4)[3:].plot(lw = 2);
ax.xaxis.set_minor_formatter(NullFormatter())
ax.xaxis.set_major_locator(MonthLocator(bymonthday=1))
ax.xaxis.set_major_formatter(DateFormatter("%b"))
ax.set_xlabel("");

# <markdowncell>

# Exercises: Pandas
# ====== 
# 
# * Using the same data as in the examples, calculate mean monthly closing prices but using only odd dates (i.e 1st of Jul, 3rd of Jul, ...)

# <markdowncell>

# # Basemap
# ## The author is Jeff Whitaker.  
# ## Basemap code is available on [github](https://github.com/matplotlib/basemap).
# 
# ## Example [gallery](http://matplotlib.org/basemap/users/examples.html).
# 
# PyNGL - can be used as an alternative.

# <markdowncell>

# # The usual workflow with the library:
# 
# 
# * **Create a basemap object**
# <pre class="co">
# basemap = Basemap(projection="...", ...)
# </pre>
# 
# 
# * **Draw your field using:** 
# <pre class="co">
# basemap.contour(), basemap.contourf(), 
# basemap.pcolormesh()
# </pre>
# 
# * **It provides utility functions to draw coastlines, meridians and shapefiles (only those that contain lat/lon coordinates), mask ocean points.**
# 
# * **Drawing the colorbar is as easy as:**
# <pre class="co">
# basemap.colorbar(img)
# </pre>
#     

# <markdowncell>

# # Basemap
# Can be used for nice map backgrounds

# <codecell>

from mpl_toolkits.basemap import Basemap
b = Basemap()
fig, (ax1, ax2) = plt.subplots(1,2)

b.warpimage(ax = ax1, scale = 0.1); 
im = b.etopo(ax = ax2, scale = 0.1); 

# <markdowncell>

# Define a helper download function
# ======

# <codecell>

import os
def download_link(url, local_path):
    if os.path.isfile(local_path):
        return
    
    import urllib2
    s = urllib2.urlopen(url)
    with open(local_path, "wb") as local_file:
        local_file.write(s.read())   

# <markdowncell>

# # Basemap: display model results in rotated lat/lon projection

# <codecell>

f_name = "monthly_mean_qc_0.1_1979_05_PR.rpn.nc"
base_url = "http://scaweb.sca.uqam.ca/~huziy/example_data"
#Fetch the file if it is not there
url = os.path.join(base_url, f_name)
download_link(url, f_name)

#read the file and check what is inside
ds_pr = Dataset(f_name)
print ds_pr.variables.keys()

# <markdowncell>

# Read data from the NetCDF file
# ====

# <codecell>

rotpole = ds_pr.variables["rotated_pole"]
lon = ds_pr.variables["lon"][:]
lat = ds_pr.variables["lat"][:]
data = ds_pr.variables["preacc"][:].squeeze()
#rotpole.ncattrs(); - returns a list of netcdf attributes of a variable
print rotpole.grid_north_pole_latitude, \
      rotpole.grid_north_pole_longitude, \
      rotpole.north_pole_grid_longitude

# <markdowncell>

# Creating a basemap object for the data
# ======

# <codecell>

lon_0 = rotpole.grid_north_pole_longitude - 180
o_lon_p = rotpole.north_pole_grid_longitude
o_lat_p = rotpole.grid_north_pole_latitude
b = Basemap(projection="rotpole", 
            lon_0=lon_0, 
            o_lon_p = o_lon_p, 
            o_lat_p = o_lat_p,
            llcrnrlon = lon[0, 0], 
            llcrnrlat = lat[0, 0],
            urcrnrlon = lon[-1, -1], 
            urcrnrlat = lat[-1, -1], 
            resolution="l")

# <markdowncell>

# Projecting to and from the projection coordinates
# =======

# <codecell>

#(lat, lon) -> (x, y)
x, y = b(lon, lat)

#(x,y) -> (lat, lon)
loninv, latinv = b(x, y, inverse=True)

# <markdowncell>

# Define plotting function to reuse
# =====

# <codecell>

def plot_data(x, y, data, ax):
    im = b.contourf(x, y, data, ax = ax)
    b.colorbar(im, ax = ax)
    b.drawcoastlines(linewidth=0.5);

# <markdowncell>

# Plot the data
# =====

# <codecell>

fig, (ax1, ax2) = plt.subplots(1,2)
#plot data as is
plot_data(x, y, data, ax1)
#mask small values
to_plot = np.ma.masked_where(data <= 2, data) 
plot_data(x, y, to_plot, ax2)

# <markdowncell>

# Masking oceans is as easy as
# =====

# <codecell>

from mpl_toolkits.basemap import maskoceans
lon1 = lon.copy()
lon1[lon1 > 180] = lon1[lon1 > 180] - 360
data_no_ocean = maskoceans(lon1, lat, data)

# <markdowncell>

# Plot masked field
# =====

# <codecell>

fig, ax = plt.subplots(1,1)
plot_data(x, y, data_no_ocean, ax)

# <markdowncell>

# # Basemap: reading and visualizing shape files (Download data)

# <codecell>

from urllib2 import urlopen
import re, os

# <codecell>

#download the shape files directory
local_folder = "countries"
remote_folder = 'http://scaweb.sca.uqam.ca/~huziy/example_data/countries/'
if not os.path.isfile(local_folder+"/cntry00.shp"):
    urlpath = urlopen(remote_folder)
    string = urlpath.read().decode('utf-8')
    pattern = re.compile(r'cntry00\...."')
    filelist = pattern.findall(string)
    filelist = [s[:-1] for s in filelist if not s.endswith('zip"')]
    
    if not os.path.isdir(local_folder):
        os.mkdir(local_folder)
    for fname in filelist:
        f_path = os.path.join(local_folder, fname)
        remote_f_path = os.path.join(remote_folder, fname)
        #download selected files
        download_link(remote_f_path, f_path)

# <markdowncell>

# # Basemap: reading and visualizing shape files

# <codecell>

bworld = Basemap()
shp_file = os.path.join(local_folder, "cntry00")
ncountries, _, _, _, linecollection = bworld.readshapefile(shp_file, "country")

# <markdowncell>

# Reading country polygons (and attributes) from the shape file, necessary imports 
# =====

# <codecell>

import fiona
from matplotlib.collections import PatchCollection
from shapely.geometry import MultiPolygon, shape
from matplotlib.patches import Polygon
import shapely
import matplotlib.cm as cm

# <markdowncell>

# Reading country polygons (and attributes) from the shape file, preparations 
# =====

# <codecell>

cmap_shape = cm.get_cmap("Greens", 10)
bounds = np.arange(0, 1.65, 0.15) * 1e9
bn_shape = BoundaryNorm(bounds, len(bounds) - 1)

bworld = Basemap()
populations = []; patches = []

def to_mpl_poly(shp_poly, apatches): 
    a = np.asarray(shp_poly.exterior)
    x, y = bworld(a[:, 0], a[:, 1])
    apatches.append(Polygon(zip(x, y)))

# <markdowncell>

# Reading country polygons (and attributes) from the shape file, preparations 
# =====

# <codecell>

with fiona.open('countries/cntry00.shp', 'r') as inp:
    for f in inp:
        the_population = f["properties"]["POP_CNTRY"]
        sh = shape(f['geometry'])
        
        if isinstance(sh, shapely.geometry.Polygon):
            to_mpl_poly(sh, patches)
            populations.append(the_population)
        elif isinstance(sh, shapely.geometry.MultiPolygon):
            for sh1 in sh.geoms:
                to_mpl_poly(sh1, patches)
                populations.append(the_population)

# <markdowncell>

# Coloring the countries based on the attribute value
# =========

# <codecell>

sf = ScalarFormatter(useMathText=True)

# <codecell>

fig = plt.figure(); ax = fig.add_subplot(111)
pcol = PatchCollection(patches, cmap = cmap_shape, norm = bn_shape)
pcol.set_array(np.array(populations))
ax.add_collection(pcol)
cb = bworld.colorbar(pcol, ticks = bounds, format = sf)
cb.ax.yaxis.get_offset_text().set_position((-2,0))
bworld.drawcoastlines(ax = ax, linewidth = 0);

# <markdowncell>

# Exercise: shape files
# ======
# 
# * There is a [script](http://scaweb.sca.uqam.ca/~huziy/example_data/fiona_test.py) which contains a bug. Copy it to your working directory and try to fix the bug. Try launching it and see if the produced image is correct.

# <markdowncell>

# # Basemap: plotting wind field (read and convert units)

# <codecell>

download_link("http://scaweb.sca.uqam.ca/~huziy/example_data/wind.nc", "wind.nc")
ds_wind = Dataset("wind.nc")
u = ds_wind.variables["UU"][:]
v = ds_wind.variables["VV"][:]
coef_wind = 0.51444444444 #m/s in one knot
ds_wind.close()

# <markdowncell>

# # Basemap: plotting wind field

# <codecell>

fig = plt.figure()
fig.set_size_inches(6,8)
Q = b.quiver(x[::8, ::8], y[::8, ::8], 
             u[::8, ::8] * coef_wind, v[::8, ::8] * coef_wind)
# make quiver key.
qk = plt.quiverkey(Q, 0.35, 0.1, 5, '5 m/s', labelpos='N',
                   coordinates = "figure", 
                   fontproperties = dict(weight="bold"))
b.drawcoastlines(linewidth = 0.1);
plt.savefig("wind.png");

# <markdowncell>

# # Basemap: plotting wind field (saved image, the legend is not cropped)
# <img src="files/images/wind.jpeg"/>

# <markdowncell>

# Exercises: Basemap
# =====

# <markdowncell>

# * Plot fields `preacc` and `air` from [here](http://scaweb.sca.uqam.ca/~huziy/example_data/wm201_Arctic_JJA_1990-2008_moyenneDesMoyennes.nc) using basemap. Add meridians and parallels as well as rivers to the map. Use subplots to put the plots side by side.
# 
# 
# * Plot field `preacc` from [here](http://scaweb.sca.uqam.ca/~huziy/example_data/pm1989010100_00000498p_0.44deg_africa_PR.nc).

# <markdowncell>

# cKDTree
# ======
# 
# * Is a class defined in ```scipy.spatial``` package.
# 
# 
# * Alternatives: ```pyresample``` and ```basemap.interp```

# <markdowncell>

# cKDTree - workflow
# ======
# 
# ##1. Convert all lat/lon coordinates to Cartesian coordinates
# <pre class="co">
#     (xs, ys, zs) #correspond to coordinates of the source grid
#     (xt, yt, zt) #coordinates of the target grid
#     #All x,y,z - are flattened 1d arrays
# </pre>
# 
# ##2. Create a cKDTree object representing source grid
# <pre class="co">
#     tree = cKDTree(data = zip(xs, ys, zs))
# </pre>

# <markdowncell>

# cKDTree - workflow
# ======
# ## 3. Query indices of the nearest target cells and distances to the corresponding source cells
# <pre class="co"> 
#     dists, inds = tree.query(zip(xt, yt, zt), k = 1)
#     #k = 1, means find 1 nearest point
# </pre>
# 
# ## 4. Get interpolated data (and reshape it to 2D)
# <pre class="co">
#     data_target = data_source[inds].reshape(lon_target.shape)
# </pre>

# <markdowncell>

# cKDTree
# ======
# 
# * For an example usage of ```cKDTree```, please read my post at [earthpy.org](http://earthpy.org/interpolation_between_grids_with_ckdtree.html) 
# 
# 
# * There is a folowup to that post about ```pyresample``` [here](http://earthpy.org/interpolation_between_grids_with_pyresample.html).

# <markdowncell>

# cKDTree: exercise
# ======
# 
# * Interpolate CRU temperatures to the model grid, use inverse distance weighting for intepolation from 20 nearest points.
# 
#     $$
#     T_t = \frac{\sum_{i=1}^{20}w_i T_{s,i}}{\sum_{i=1}^{20}w_i}
#     $$ 
# 
#     where $w_i = 1/d_i^2$, $d_i$ - is the distance between corresponding source and target grid points.
#     Grid longitudes and latitudes are defined in this [file](http://scaweb.sca.uqam.ca/~huziy/example_data/wind.nc). The observation temperature field is [here](http://scaweb.sca.uqam.ca/~huziy/example_data/mean_temp_cru.nc).

# <markdowncell>

# Speeding up your code
# =====
# 
# * Avoid loops and try to make use of Numpy/Scipy/Pandas functionality as much as possible
# 
# 
# * If you have a lot of computations the speedup can be achieved by delegating the computations to the NumExpr module.
# 
# 
# * You can use Cython to compile libraries to get C-like speed.
# 
# 
# * Use multiprocessing module if the tasks are fairly independent and cannot be solved using the above methods

# <markdowncell>

# Multiprocessing example
# ====

# <codecell>

from multiprocessing import Pool

def func(x):
    return x ** 2

p1 = Pool(processes=1)
p2 = Pool(processes=2)

nums = np.arange(100000)

# <markdowncell>

# Make sure it works correctly
# ----

# <codecell>

print p2.map(func, nums)[:10]

# <markdowncell>

# Tests of performance gain by using 2 processes instead of 1
# ----

# <codecell>

#%%timeit 

p1.map(func, nums)

# <codecell>

#%%timeit 

p2.map(func, nums)

# <markdowncell>

# Squaring example using Numpy
# ===

# <codecell>

#%%timeit
sq = np.asarray(nums) ** 2

# <markdowncell>

# Squaring example using NumExpr
# ===

# <codecell>

import numexpr as ne
print ne.detect_number_of_cores()
print ne.evaluate("nums ** 2")[:5]

# <codecell>

#%%timeit
ne.evaluate("nums ** 2")

# <codecell>

ne.set_num_threads(1)

# <codecell>

#%%timeit
ne.evaluate("nums**2")

# <markdowncell>

# Squaring example using cythonized function
# ===

# <codecell>

#%load_ext cythonmagic

# <codecell>

#%%cython
def square_list(the_list):
    result = []
    cdef int i
    for i in the_list:
        result.append(i ** 2)
    return result

# <codecell>

#%%timeit
square_list(nums)

# <markdowncell>

# Resources
# =========
# 
# * Python: [http://www.python.org/](http://www.python.org/)
# 
# 
# * IPython: [http://ipython.org/](http://ipython.org/)
# 
# 
# * NumPy: [http://www.numpy.org/](http://www.numpy.org/) - there is a page NumPy for MATLAB users, might be useful for those familiar with MATLAB.
# 
# 
# * SciPy: [http://www.scipy.org/](http://www.scipy.org/)
# 
# 
# * Scientific python lectures: [http://scipy-lectures.github.io/](http://scipy-lectures.github.io/)
# 
# 
# * Matplotlib: [http://matplotlib.org/](http://matplotlib.org/)
# 
# 
# * Scitools (iris and cartopy): [http://scitools.org.uk/index.html](http://scitools.org.uk/index.html)

# <markdowncell>

# Resources
# =========
# 
# * NetCDF4: [http://netcdf4-python.googlecode.com/svn/trunk/docs/netCDF4-module.html](http://netcdf4-python.googlecode.com/svn/trunk/docs/netCDF4-module.html)
# 
# 
# * Basemap: [http://matplotlib.org/basemap/](http://matplotlib.org/basemap/)
# 
# 
# * [Notebook on shape files](http://nbviewer.ipython.org/github/mqlaql/geospatial-data/blob/master/Geospatial-Data-with-Python.ipynb)
# 
# 
# * My favourite python IDE: [Pycharm](http://www.jetbrains.com/pycharm/), you might also checkout many other like Eclipse, Netbeans, Spyder...
# 
# 
# * On creating presentations using IPython and a lot of other things: [Damian Avilla's blog](http://www.damian.oquanta.info/index.html).

# <codecell>


