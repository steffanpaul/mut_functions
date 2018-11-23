#!/usr/bin/env python

import os
from subprocess import call
import sys
from shutil import copyfile

copylist = ['mutagenesisfunctions.py', 'helper.py', 'bpdev.py']
#now copy them back one folder
for f in copylist:
    copyfile(f, os.path.join('..', f))
