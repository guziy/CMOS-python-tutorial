#!/bin/bash

ipython nbconvert cmos2014-python-tutorial.ipynb --to slides --post serve --config slides_config.py --template default_transition.tpl
