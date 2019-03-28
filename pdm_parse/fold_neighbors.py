import numpy as np
from glob import glob
import copy
import itertools
from shutil import copyfile
import sys
import os
import subprocess as sp

from astropy.coordinates import SkyCoord
from astropy import units as u


#########################
## CANDIDATE CLASS DEF ##
#########################

class Candidate(object):
    def __init__(self, line, beam):
        self.beam     = beam
        cols          = line.split()
        self.filename = cols[0].split(":")[0]
        self.filecand = int( cols[0].split(":")[1] )
        self.DM       = float(cols[1])
        self.SNR      = float(cols[2])
        self.sigma    = float(cols[3])
        self.numharm  = int(cols[4])
        self.ipow     = float(cols[5])
        self.cpow     = float(cols[6])
        self.P        = float(cols[7])
        self.r        = float(cols[8])
        self.z        = float(cols[9])
        self.numhits  = int( cols[10].strip("()") )
        self.line     = line
        self.note     = ''

        self.h_hits  = 0
        self.h_harms = []
        self.h_Ps    = []
        self.h_DMs   = []
        self.h_SNRs  = []
        self.h_beams = []

    def add_harm(self, cand):
        self.h_hits += 1
        self.h_harms.append( self.P / cand.P )
        self.h_Ps.append( cand.P )
        self.h_DMs.append( cand.DM )
        self.h_SNRs.append( cand.SNR )
        self.h_beams.append( cand.beam )

    def __repr__(self):
        repr_string = "Cand(beam=%d, SNR=%.0f, P=%.2f, DM=%1f)" \
            %(self.beam, self.SNR, self.P, self.DM)
        return repr_string


def cmp_snr(a, b):
    return -1 * cmp(a.SNR, b.SNR)


def cmp_sigma(a, b):
    return -1 * cmp(a.sigma, b.sigma)


##################
## CAND PARSING ##
##################

def cands_from_file(candfile, beam):
    candlist = []
    with open(candfile, 'r') as fin:
        for line in fin:
            if line[0] in ["#", " ", "\n"]:
                continue
            else:
                pass
            candlist.append( Candidate(line, beam) )
    return candlist


def cands_from_many_files(indir, name_tmp, beams):
    candlist = []
    for bb in beams:
        fname  = name_tmp % bb
        infile = "%s/%s" %(indir, fname)
        if os.path.isfile(infile):
            print("Beam %d" %(bb))
            cc = cands_from_file(infile, bb)
            candlist += cc
        else:
            print("   Beam %d File Missing" %(bb))
    
    candlist.sort(cmp=cmp_sigma)
    return candlist


####################
## FILE SHUTTLING ##
####################

def filename_from_cand(cand, suffix='png'):
    fname = cand.filename.rstrip('_100')
    fnum  = cand.filecand
    return "%s_Cand_%d.pfd.%s" %(fname, fnum, suffix)


def copy_cand_plots(candlist, topdir, outdir, suffix='png'):
    for ii, cc in enumerate(candlist):
        fname = filename_from_cand(cc, suffix=suffix)
        out_fname = "cand%05d_%s" %(ii, fname)
        infile  = "%s/beam%05d/cands_presto/%s" %(topdir, cc.beam, fname)
        outfile = "%s/%s" %(outdir, out_fname)
        copyfile(infile, outfile)
    return


####################
## BEAM NEIGHBORS ##
####################

def get_skycoords(beams):
    ra_strs  = beams[:, 1]
    dec_strs = beams[:, 2]
    dec_strs = np.array([ dd.replace('.', ':', 2) for dd in dec_strs ])
    coords = SkyCoord(ra=ra_strs, dec=dec_strs, unit=(u.hour, u.deg))
    return coords


def get_beam_offsets(coords, beams, coord0=None):
    cc = coords[beams]
    if coord0 is None:
        coord0 = cc[0]
    else:
        pass
    ra_offset = ((cc.ra - coord0.ra) * np.cos(coord0.dec.to('radian'))).arcsec
    dec_offset = (cc.dec - coord0.dec).arcsec
    return ra_offset, dec_offset


def get_neighbors(coords, beams, theta, beam0, level=1):
    cc = coords
    idx = np.where(beams == beam0)[0]
    if not len(idx):
        print("Beam not in list!!")
        return np.array([beam0])
    else: 
        pass
    idx0 = idx[0]
    coord0 = cc[idx0]
    dra  = ((cc.ra - coord0.ra) * np.cos(coord0.dec.to('radian'))).arcsec
    ddec = (cc.dec - coord0.dec).arcsec
    dd   = np.sqrt(dra**2.0 + ddec**2.0)
    xx = np.where( dd < (level+0.1) * theta )[0]
    return beams[xx]


####################
## FOLD NEW CANDS ##
####################

def try_cmd(cmd, stdout=None, stderr=None):
    """
    Run the command in the string cmd using sp.check_call().  If there
    is a problem running, a CalledProcessError will occur and the
    program will quit.
    """
    print "\n\n %s \n\n" %cmd
    try:
        retval = sp.check_call(cmd, shell=True, stdout=stdout, stderr=stderr)
    except sp.CalledProcessError:
        print("The command:\n %s \ndid not work!!!" %cmd)
        #sys.exit(0)


def parse_name(namecand):
    namesplit = namecand.rsplit("_", 1)
    beamstr = namesplit[0].split('_')[0]
    beamnum = int(beamstr.lstrip('beam'))
    zval, candnum = namesplit[1].split(':')
    zval = int(zval)
    candnum = int(candnum)
    candfile = "%s.cand" %(namecand.split(':')[0])
    return beamstr, beamnum, zval, candnum, candfile


def read_top_cands(infile):
    """
    Read cands into dict
    """
    num = 0
    cand_list = []
    with open(infile) as fin:
        for line in fin:
            if line[0] == "#":
                continue
            else: pass

            cols = line.split()
            namecand = cols[0]
            bstr, bnum, zval, cnum, cfile = parse_name(namecand)
            dm = float(cols[1])
            cdict = {'bstr' : bstr,
                     'bnum' : bnum, 
                     'zval' : zval, 
                     'cnum' : cnum, 
                     'cfile': cfile,
                     'dm'   : dm, 
                     'num'  : num}
            cand_list.append(cdict)
            num += 1
    return cand_list


def run_prepfold_topcands(clist, pdir, top_dir, zmax, 
                          outfile, errfile, searchtype='none', 
                          extra_args='', use_mask=False):
    """
    This function will run prepfold on the candidate files produced
    by the presto accelsearch.  This essentially replaces the
    gotocand.py script from the older version of the pipeline

    Work in beam dir  (bdir)
    Fits files in pdir/part[1-7]/psrfits originally, 
    dlink over to bdir
    """
    if searchtype == 'fine':
        sflag = "-fine"
    elif searchtype =='coarse':
        sflag = "-coarse"
    elif searchtype == 'regular':
        sflag = ""
    else:
        sflag = "-nosearch"

    for cc in clist:
        bstr  = cc['bstr']
        zval  = cc['zval']
        cnum  = cc['cnum']
        cfile = cc['cfile']
        dm    = cc['dm']
        num   = cc['num']

        psname = "cand%04d_%s_DM%.2f_ACCEL_Cand_%d.pfd.ps"\
                 %(num, bstr, dm, cnum) 
        if os.path.exists(psname):
            print "File "+psname+" already made, skipping"
            continue
        else:
            pass

        bdir = "%s/%s" %(top_dir, bstr)

        # Symlink FITS files to bdir
        fitsglob = "%s/part*/psrfits/*%s*fits" %(pdir, bstr)
        fitslist = glob(fitsglob)
        fitslist.sort()

        if len(fitslist):
            fitsnames = [ fits_ii.split('/')[-1] for fits_ii in fitslist ]
            newpaths = [ "%s/%s" %(bdir, fname) for fname in fitsnames ]
            for ii in xrange(len(fitsnames)):
                orig_fits = fitslist[ii]
                link_fits = newpaths[ii]
                print("%s ---> %s" %(orig_fits, link_fits))
                os.symlink(orig_fits, link_fits)
        else:
            print("No FITS files found with %s" %fitsglob)
            continue

        fitsfiles = "%s/*fits" %(bdir)
        canddir = "%s/cands_presto" %bdir
        infdir  = "%s/dedisperse" %bdir

        if use_mask:
            maskfile = "%s/rfi_products/*.mask" %bdir
        else:
            maskfile = ''
        
        # Need to copy cand file from canddir to bdir
        orig_candfile = "%s/%s" %(canddir, cfile)
        candfile = "%s/%s" %(bdir, cfile)
        copyfile(orig_candfile, candfile)
        print(orig_candfile)
        print(candfile)

        # UGH: Need to copy inf file too for some reason
        inf_name = "%s_DM%.2f.inf" %(bstr, dm)
        orig_inf_file = "%s/dedisperse/%s" %(bdir, inf_name)
        inf_file = "%s/%s" %(bdir, inf_name)
        copyfile(orig_inf_file, inf_file)
        print(orig_inf_file)
        print(inf_file)

        # Now we can run prepfold command
        outname = "cand%04d_%s_z%d" %(num, bstr, zval)
        if ( os.path.exists(candfile) ):
            if use_mask:
                cmd = "prepfold -noxwin %s -dm %.2f " %(sflag, dm) +\
                      "-accelcand %d -accelfile %s " %(cnum, candfile) +\
                      "%s " %(extra_args) +\
                      "-noweights -noscales -nooffsets " +\
                      "-mask %s -o %s %s" %(maskfile, outname, fitsfiles)
            else:
                cmd = "prepfold -noxwin %s -dm %.2f " %(sflag, dm) +\
                      "-accelcand %d -accelfile %s " %(cnum, candfile) +\
                      "%s " %(extra_args) +\
                      "-noweights -noscales -nooffsets " +\
                      "-o %s %s" %(outname, fitsfiles)
        
            try_cmd(cmd, stdout=outfile, stderr=errfile)

        else:
            print "Could not find %s" %candfile
            print "and/or         %s" %datfile

        # Now we remove the copied files
        os.remove(candfile)
        os.remove(inf_file)

        # Unlink the fits files
        if len(fitslist):
            for lfits in newpaths:
                os.unlink(lfits)
        else: pass

    return


def copy_beam_ddm(bnum, DM, search_dir, out_dir):
    """
    Copy *dat and *inf files for a given DM
    assuming a file structure of 
    
      %s/beam%05d/dedisperse/

    and a file name like:

      beam00200_DM3180.00.dat
    """
    bstr = "beam%05d" %(bnum)
    ddm_glob = "%s/%s/dedisperse/%s_DM%.2f.*[dat,inf]" \
                %(search_dir, bstr, bstr, DM)
    print(ddm_glob)
    ddms = glob(ddm_glob)
    if not len(ddms):
        print("No Files found!!!")
        return
    else:
        pass

    for ddm_path in ddms:
        fname = ddm_path.split('/')[-1]
        out_path = "%s/%s" %(out_dir, fname)
        print(ddm_path)
        print(out_path)
        copyfile(ddm_path, out_path)

    return


def clean_up(indir, ext_list):
    for ext in ext_list:
        ext_files = glob("%s/*%s" %(indir, ext))
        for efile in ext_files:
            print("Removing: %s" %efile)
            os.remove(efile)
    return


def run_prepfold_neighbors(clist, coords, beams, theta, 
                          level, search_dir, out_dir,
                          outfile=None, errfile=None, 
                          extra_args=''):
    """
    Find adjacent beams to cand beam, fold 'em all
    with center beam cand info
    """
    sflag = "-nosearch"

    # Go to results directory
    os.chdir(out_dir)

    clean_ext = ["dat", "inf"]

    for cc in clist:
        bstr  = cc['bstr']
        bnum  = cc['bnum']
        zval  = cc['zval']
        cnum  = cc['cnum']
        cfile = cc['cfile']
        dm    = cc['dm']
        num   = cc['num']

        cdir = "%s/Cand%04d" %(out_dir, num)
        
        # If cdir doesn't exist, make it 
        # If it does, skip processing
        if not os.path.exists(cdir):
            os.makedirs(cdir)
        else:
            print("%s already exists, skipping!" %cdir)
            continue

        # Get the beam numbers we need
        n_beams = get_neighbors(coords, beams, theta, bnum, level=level) 
        print(n_beams)

        # Need to copy cand file from canddir to bdir
        orig_candfile = "%s/%s/cands_presto/%s" %(search_dir, bstr, cfile)
        candfile = "%s/%s" %(cdir, cfile)
        copyfile(orig_candfile, candfile)
        print(orig_candfile)
        print(candfile)

        # Copy over all dat and inf files now
        for n_bnum in n_beams:
            copy_beam_ddm(n_bnum, dm, search_dir, cdir)

        # Now we can run prepfold command for each beam
        for n_bnum in n_beams:
            # Set dat file and outfile names
            datfile = "Cand%04d/beam%05d_DM%.2f.dat" %(num, n_bnum, dm)
            inffile = "Cand%04d/beam%05d_DM%.2f.inf" %(num, n_bnum, dm)
      
            # Assumes we are in out_dir
            outname = "Cand%04d/beam%05d" %(num, n_bnum)

            if ( os.path.exists(candfile) ):
                cmd = "prepfold -noxwin "  +\
                      "-accelcand %d -accelfile %s " %(cnum, candfile) +\
                      "%s %s " %(sflag, extra_args) +\
                      "-o %s %s" %(outname, datfile)
        
                try_cmd(cmd, stdout=outfile, stderr=errfile)

            else:
                print "Could not find %s" %candfile
                print "and/or         %s" %datfile

        
        clean_up(cdir, clean_ext)

        # Can now remove candfile
        if os.path.exists(candfile):
            os.remove(candfile)

    return


##############
###  MISC  ###
##############

def attrarr(obj_list, attr):
    if hasattr(obj_list[0], attr):
        out_arr = np.array([ getattr(bb, attr) for bb in obj_list ])
        return out_arr
    else:
        print("List has no attribute \"%s\" " %attr)
        return



############
### MAIN ###
############
#sys.exit()

group = '57519'

top_dir = "/hercules/results/rwharton/fastvis_gc/proc/%s" %(group)
search_dir = "%s/search" %(top_dir)
out_dir = "%s/search/neighbors" %(top_dir)

beamfile = '/hercules/results/rwharton/fastvis_gc/proc/gc_beamlist_hex.npy'
#beamfile = "gc_beamlist_hex.npy"
beam_locs = np.load(beamfile)
beam_nums = np.arange(0, 1261)
beam_locs = beam_locs[beam_nums]
coords = get_skycoords(beam_locs)
theta = 3.0
level = 2

top_name = "%s/good_cands.txt" %(out_dir)

clist = read_top_cands(top_name)
#clist = clist[:1]
run_prepfold_neighbors(clist, coords, beam_nums, theta, level, search_dir, out_dir)

