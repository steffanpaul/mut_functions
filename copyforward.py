#!/usr/bin/env python

import os
from subprocess import call
import sys
from shutil import copyfile

copylist = ['mutagenesisfunctions.py', 'helper.py', 'bpdev.py']
#now copy them forward into the mut_functions folder
for f in copylist:
    copyfile(os.path.join('..', f), f)
