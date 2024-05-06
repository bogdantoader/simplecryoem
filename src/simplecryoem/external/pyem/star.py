# Copyright (C) 2016 Daniel Asarnow
# University of California, San Francisco
#
# Library for parsing and altering Relion .star files.
# See help text and README file for more information.
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
from __future__ import print_function
import os.path
import pandas as pd
import sys


class Relion:
    # Relion 2+ fields.
    MICROGRAPH_NAME = "rlnMicrographName"
    MICROGRAPH_NAME_NODW = "rlnMicrographNameNoDW"
    IMAGE_NAME = "rlnImageName"
    IMAGE_ORIGINAL_NAME = "rlnImageOriginalName"
    RECONSTRUCT_IMAGE_NAME = "rlnReconstructImageName"
    COORDX = "rlnCoordinateX"
    COORDY = "rlnCoordinateY"
    ORIGINX = "rlnOriginX"
    ORIGINY = "rlnOriginY"
    ORIGINZ = "rlnOriginZ"
    ANGLEROT = "rlnAngleRot"
    ANGLETILT = "rlnAngleTilt"
    ANGLEPSI = "rlnAnglePsi"
    CLASS = "rlnClassNumber"
    DEFOCUSU = "rlnDefocusU"
    DEFOCUSV = "rlnDefocusV"
    DEFOCUS = [DEFOCUSU, DEFOCUSV]
    DEFOCUSANGLE = "rlnDefocusAngle"
    CS = "rlnSphericalAberration"
    PHASESHIFT = "rlnPhaseShift"
    AC = "rlnAmplitudeContrast"
    VOLTAGE = "rlnVoltage"
    MAGNIFICATION = "rlnMagnification"
    DETECTORPIXELSIZE = "rlnDetectorPixelSize"
    BEAMTILTX = "rlnBeamTiltX"
    BEAMTILTY = "rlnBeamTiltY"
    BEAMTILTCLASS = "rlnBeamTiltClass"
    CTFSCALEFACTOR = "rlnCtfScalefactor"
    CTFBFACTOR = "rlnCtfBfactor"
    CTFMAXRESOLUTION = "rlnCtfMaxResolution"
    CTFFIGUREOFMERIT = "rlnCtfFigureOfMerit"
    GROUPNUMBER = "rlnGroupNumber"
    RANDOMSUBSET = "rlnRandomSubset"
    AUTOPICKFIGUREOFMERIT = "rlnAutopickFigureOfMerit"

    # Relion 3 fields.
    OPTICSGROUP = "rlnOpticsGroup"
    OPTICSGROUPNAME = "rlnOpticsGroupName"
    ODDZERNIKE = "rlnOddZernike"
    EVENZERNIKE = "rlnEvenZernike"
    MAGMAT00 = "rlnMagMat00"
    MAGMAT01 = "rlnMagMat01"
    MAGMAT10 = "rlnMagMat10"
    MAGMAT11 = "rlnMagMat11"
    IMAGEPIXELSIZE = "rlnImagePixelSize"
    IMAGESIZE = "rlnImageSize"
    IMAGEDIMENSION = "rlnImageDimensionality"
    ORIGINXANGST = "rlnOriginXAngst"
    ORIGINYANGST = "rlnOriginYAngst"
    ORIGINZANGST = "rlnOriginZAngst"
    MICROGRAPHPIXELSIZE = "rlnMicrographPixelSize"
    MICROGRAPHORIGINALPIXELSIZE = "rlnMicrographOriginalPixelSize"
    MTFFILENAME = "rlnMtfFileName"

    # Field lists.
    COORDS = [COORDX, COORDY]
    ORIGINS = [ORIGINX, ORIGINY]
    ORIGINS3D = [ORIGINX, ORIGINY, ORIGINZ]
    ORIGINSANGST = [ORIGINXANGST, ORIGINYANGST]
    ORIGINSANGST3D = [ORIGINXANGST, ORIGINYANGST, ORIGINZANGST]
    ANGLES = [ANGLEROT, ANGLETILT, ANGLEPSI]
    ALIGNMENTS = ANGLES + ORIGINS3D + ORIGINSANGST3D
    CTF_PARAMS = [DEFOCUSU, DEFOCUSV, DEFOCUSANGLE, CS, PHASESHIFT, AC,
                  BEAMTILTX, BEAMTILTY, BEAMTILTCLASS, CTFSCALEFACTOR, CTFBFACTOR,
                  CTFMAXRESOLUTION, CTFFIGUREOFMERIT]
    MICROSCOPE_PARAMS = [VOLTAGE, MAGNIFICATION, DETECTORPIXELSIZE]
    MICROGRAPH_COORDS = [MICROGRAPH_NAME] + COORDS
    PICK_PARAMS = MICROGRAPH_COORDS + [ANGLEPSI, CLASS, AUTOPICKFIGUREOFMERIT]

    FIELD_ORDER = [IMAGE_NAME, IMAGE_ORIGINAL_NAME, MICROGRAPH_NAME, MICROGRAPH_NAME_NODW] + \
        COORDS + ALIGNMENTS + MICROSCOPE_PARAMS + CTF_PARAMS + \
                  [CLASS + GROUPNUMBER + RANDOMSUBSET + OPTICSGROUP]

    RELION2 = ORIGINS3D + [MAGNIFICATION, DETECTORPIXELSIZE]

    RELION30 = [BEAMTILTCLASS]

    RELION31 = ORIGINSANGST3D + [BEAMTILTX, BEAMTILTY, OPTICSGROUP, OPTICSGROUPNAME,
                                 ODDZERNIKE, EVENZERNIKE, MAGMAT00, MAGMAT01, MAGMAT10, MAGMAT11,
                                 IMAGEPIXELSIZE, IMAGESIZE, IMAGEDIMENSION]

    OPTICSGROUPTABLE = [AC, CS, VOLTAGE, BEAMTILTX, BEAMTILTY, OPTICSGROUPNAME, ODDZERNIKE, EVENZERNIKE,
                        MAGMAT00, MAGMAT01, MAGMAT10, MAGMAT11, IMAGEPIXELSIZE, IMAGESIZE, IMAGEDIMENSION]

    # Data tables.
    OPTICDATA = "data_optics"
    MICROGRAPHDATA = "data_micrographs"
    PARTICLEDATA = "data_particles"
    IMAGEDATA = "data_images"


class UCSF:
    IMAGE_PATH = "ucsfImagePath"
    IMAGE_BASENAME = "ucsfImageBasename"
    IMAGE_INDEX = "ucsfImageIndex"
    IMAGE_ORIGINAL_PATH = "ucsfImageOriginalPath"
    IMAGE_ORIGINAL_BASENAME = "ucsfImageOriginalBasename"
    IMAGE_ORIGINAL_INDEX = "ucsfImageOriginalIndex"
    MICROGRAPH_BASENAME = "ucsfMicrographBasename"
    UID = "ucsfUid"
    PARTICLE_UID = "ucsfParticleUid"
    MICROGRAPH_UID = "ucsfMicrographUid"


def calculate_apix(df):
    try:
        if df.ndim == 2:
            if Relion.IMAGEPIXELSIZE in df:
                return df.iloc[0][Relion.IMAGEPIXELSIZE]
            if Relion.MICROGRAPHPIXELSIZE in df:
                return df.iloc[0][Relion.MICROGRAPHPIXELSIZE]
            return 10000.0 * df.iloc[0][Relion.DETECTORPIXELSIZE] / df.iloc[0][Relion.MAGNIFICATION]
        elif df.ndim == 1:
            if Relion.IMAGEPIXELSIZE in df:
                return df[Relion.IMAGEPIXELSIZE]
            if Relion.MICROGRAPHPIXELSIZE in df:
                return df[Relion.MICROGRAPHPIXELSIZE]
            return 10000.0 * df[Relion.DETECTORPIXELSIZE] / df[Relion.MAGNIFICATION]
        else:
            raise ValueError
    except KeyError:
        return None


def parse_star_table(starfile, offset=0, nrows=None, keep_index=False):
    headers = []
    foundheader = False
    ln = 0
    with open(starfile, 'r') as f:
        f.seek(offset)
        for l in f:
            if l.lstrip().startswith("_"):
                foundheader = True
                lastheader = True
                if keep_index:
                    head = l.strip()
                else:
                    head = l.split('#')[0].strip().lstrip('_')
                headers.append(head)
            else:
                lastheader = False
            if foundheader and not lastheader:
                break
            ln += 1
        f.seek(offset)
        df = pd.read_csv(f, delimiter='\s+', header=None, skiprows=ln, nrows=nrows)
    df.columns = headers
    return df


def star_table_offsets(starfile):
    tables = {}
    with open(starfile) as f:
        l = f.readline()  # Current line
        ln = 0  # Current line number.
        offset = 0  # Char offset of current table.
        cnt = 0  # Number of tables.
        in_table = False  # True if file cursor is inside a table.
        in_loop = False
        blank_terminates = False
        while l:
            if l.startswith("data"):
                table_name = l.strip()
                if in_table:
                    tables[table_name] = (offset, lineno, ln - 1, ln - data_line - 1)
                in_table = True
                in_loop = False
                blank_terminates = False
                offset = f.tell()  # Record byte offset of table.
                lineno = ln  # Record start line of table.
                cnt += 1  # Increment table count.
            if l.startswith("loop"):
                in_loop = True
            elif in_loop and not l.startswith("_"):
                in_loop = False
                blank_terminates = True
                data_line = ln
            if blank_terminates and in_table and l.isspace():  # Allow blankline to terminate table.
                in_table = False
                tables[table_name] = (offset, lineno, ln - 1, ln - data_line)
            l = f.readline()  # Read next line.
            ln += 1  # Increment line number.
        if in_table and table_name not in tables:
            tables[table_name] = (offset, lineno, ln, ln - data_line)
        return tables


def parse_star(starfile, keep_index=False, augment=True, nrows=sys.maxsize):
    tables = star_table_offsets(starfile)
    dfs = {t: parse_star_table(starfile, offset=tables[t][0], nrows=min(tables[t][3], nrows), keep_index=keep_index)
           for t in tables}
    if Relion.OPTICDATA in dfs:
        if Relion.PARTICLEDATA in dfs:
            data_table = Relion.PARTICLEDATA
        elif Relion.MICROGRAPHDATA in dfs:
            data_table = Relion.MICROGRAPHDATA
        elif Relion.IMAGEDATA in dfs:
            data_table = Relion.IMAGEDATA
        else:
            data_table = None
        if data_table is not None:
            df = pd.merge(dfs[Relion.OPTICDATA], dfs[data_table], on=Relion.OPTICSGROUP)
        else:
            df = dfs[Relion.OPTICDATA]
    else:
        df = dfs[next(iter(dfs))]
    df = check_defaults(df, inplace=True)
    if augment:
        augment_star_ucsf(df, inplace=True)
    return df


def augment_star_ucsf(df, inplace=True):
    df = df if inplace else df.copy()
    df.reset_index(inplace=True)
    if Relion.IMAGE_NAME in df:
        df[[UCSF.IMAGE_INDEX, UCSF.IMAGE_PATH]] = \
            df[Relion.IMAGE_NAME].str.split("@", n=2, expand=True)
        df[UCSF.IMAGE_INDEX] = pd.to_numeric(df[UCSF.IMAGE_INDEX]) - 1

        if Relion.IMAGE_ORIGINAL_NAME not in df:
            df[Relion.IMAGE_ORIGINAL_NAME] = df[Relion.IMAGE_NAME]

    if Relion.IMAGE_ORIGINAL_NAME in df:
        df[[UCSF.IMAGE_ORIGINAL_INDEX, UCSF.IMAGE_ORIGINAL_PATH]] = \
            df[Relion.IMAGE_ORIGINAL_NAME].str.split("@", n=2, expand=True)
        df[UCSF.IMAGE_ORIGINAL_INDEX] = pd.to_numeric(df[UCSF.IMAGE_ORIGINAL_INDEX]) - 1

    if UCSF.IMAGE_PATH in df:
        df[UCSF.IMAGE_BASENAME] = df[UCSF.IMAGE_PATH].apply(os.path.basename)

    if UCSF.IMAGE_ORIGINAL_PATH in df:
        df[UCSF.IMAGE_ORIGINAL_BASENAME] = df[UCSF.IMAGE_ORIGINAL_PATH].apply(os.path.basename)

    if Relion.MICROGRAPH_NAME in df:
        df[UCSF.MICROGRAPH_BASENAME] = df[Relion.MICROGRAPH_NAME].apply(os.path.basename)
    return df


def check_defaults(df, inplace=False):
    df = df if inplace else df.copy()
    if Relion.PHASESHIFT not in df:
        df[Relion.PHASESHIFT] = 0

    if Relion.IMAGEPIXELSIZE in df:
        if Relion.DETECTORPIXELSIZE not in df and Relion.MAGNIFICATION not in df:
            df[Relion.DETECTORPIXELSIZE] = df[Relion.IMAGEPIXELSIZE]
            df[Relion.MAGNIFICATION] = 10000
        elif Relion.DETECTORPIXELSIZE in df:
            df[Relion.MAGNIFICATION] = df[Relion.DETECTORPIXELSIZE] / df[Relion.IMAGEPIXELSIZE] * 10000
        elif Relion.MAGNIFICATION in df:
            df[Relion.DETECTORPIXELSIZE] = df[Relion.MAGNIFICATION] * df[Relion.IMAGEPIXELSIZE] / 10000
    elif Relion.DETECTORPIXELSIZE in df and Relion.MAGNIFICATION in df:
        # df[Relion.IMAGEPIXELSIZE] = df[Relion.DETECTORPIXELSIZE] * df[Relion.MAGNIFICATION] / 10000
        df[Relion.IMAGEPIXELSIZE] = df[Relion.DETECTORPIXELSIZE] / df[Relion.MAGNIFICATION] * 10000

    for it in zip(Relion.ORIGINSANGST3D, Relion.ORIGINS3D):
        if it[0] in df:
            df[it[1]] = df[it[0]] / df[Relion.IMAGEPIXELSIZE]
        elif it[1] in df:
            df[it[0]] = df[it[1]] * df[Relion.IMAGEPIXELSIZE]

    if Relion.ORIGINZANGST in df:
        df[Relion.IMAGEDIMENSION] = 3
    else:
        df[Relion.IMAGEDIMENSION] = 2

    if Relion.OPTICSGROUPNAME in df and Relion.OPTICSGROUP not in df:
        df[Relion.OPTICSGROUP] = df[Relion.OPTICSGROUPNAME].astype('category').cat.codes

    if Relion.BEAMTILTCLASS in df and Relion.OPTICSGROUP not in df:
        df[Relion.OPTICSGROUP] = df[Relion.BEAMTILTCLASS]
    return df
