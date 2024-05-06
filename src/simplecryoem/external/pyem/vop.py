# Copyright (C) 2017 Daniel Asarnow
# University of California, San Francisco
#
# Library functions for volume data.
# See README file for more information.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import numpy as np


def grid_correct(vol, pfac=2, order=1):
    n = vol.shape[0]
    nhalf = n // 2
    x, y, z = np.meshgrid(*[np.arange(-nhalf, nhalf)] * 3, indexing="xy")
    r = np.sqrt(x**2 + y**2 + z**2, dtype=vol.dtype) / (n * pfac)
    with np.errstate(divide="ignore", invalid="ignore"):
        sinc = np.sin(np.pi * r) / (np.pi * r)  # Results in 1 NaN in the center.
    sinc[nhalf, nhalf, nhalf] = 1.
    if order == 0:
        cordata = vol / sinc
    elif order == 1:
        cordata = vol / sinc**2
    else:
        raise NotImplementedError("Only nearest-neighbor and trilinear grid corrections are available")
    return cordata
