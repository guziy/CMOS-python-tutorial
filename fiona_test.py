#Fix a bug!!!!!


import os
import numpy as np
import matplotlib.pyplot as plt
import fiona
from descartes import PolygonPatch
from matplotlib.collections import PatchCollection
from shapely.geometry import MultiPolygon, shape, Polygon
from mpl_toolkits.basemap import Basemap

from matplotlib import cm

def download_folder():
    #download the shape files directory
    from urllib2 import urlopen
    import re

    local_folder = "countries"
    remote_folder = 'http://scaweb.sca.uqam.ca/~huziy/example_data/countries/'
    if not os.path.isdir(local_folder):
        urlpath = urlopen(remote_folder)
        string = urlpath.read().decode('utf-8')
        pattern = re.compile(r'cntry00\...."')
        filelist = pattern.findall(string)
        filelist = [s[:-1] for s in filelist if not s.endswith('zip"')]
    
        os.mkdir(local_folder)
        for fname in filelist:
            f_path = os.path.join(local_folder, fname)
            remote_f_path = os.path.join(remote_folder, fname)
            #download selected files
            download_link(remote_f_path, f_path)

def download_link(url, local_path):
    import os
    if os.path.isfile(local_path):
        return
    
    import urllib2
    s = urllib2.urlopen(url)
    with open(local_path, "wb") as local_file:
        local_file.write(s.read())   

if __name__ == "__main__":
    download_folder()

    polygons = []
    populations = []  
    with fiona.open('countries/cntry00.shp', 'r') as inp:
        for f in inp:
            the_population = f["properties"]["POP_CNTRY"]
            try:
                sh = shape(f["geometry"])
                assert sh.is_valid
                if isinstance(sh, Polygon):
                    polygons.append(PolygonPatch(sh))
                    populations.append(the_population)
                else:
                    mp = MultiPolygon(sh)
                    for pol in mp:
                        polygons.append(PolygonPatch(pol))
            except TypeError, err:
                print type(sh)
                raise err
            
        
    populations = np.array(populations)
    cmap = cm.get_cmap("Dark2", 20)

    print len(polygons)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    bworld = Basemap()
    pcol = PatchCollection(polygons, cmap = cmap)
    pcol.set_array(populations)
    ax.add_collection(pcol)
    bworld.colorbar(pcol)
    bworld.drawcoastlines()
    plt.show()
