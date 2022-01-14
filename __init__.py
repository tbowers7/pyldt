# -*- coding: utf-8 -*-
#
#  This file is part of PyLDT.
#
#   This Source Code Form is subject to the terms of the Mozilla Public
#   License, v. 2.0. If a copy of the MPL was not distributed with this
#   file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#  Created on 27-Oct-2020
#
#  @author: tbowers

"""Init File
"""

# Boilerplate variables
__author__ = 'Timothy P. Ellsworth Bowers'
__copyright__ = 'Copyright 2021'
__credits__ = ['Lowell Observatory']
__license__ = 'MPL-2.0'
__version__ = '0.2.0'
__email__ = 'tbowers@lowell.edu'
__status__ = 'Development Status :: 3 - Alpha'


# Import the user-facing functions to make them available under pyldt
from .pyldt.astrometry import solve_field
from .pyldt.imutils import *
from .pyldt.reduction import LMI, DeVeny
from .pyldt.twodspec import *
