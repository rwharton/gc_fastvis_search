import os
import sys
import time
import shlex
from glob import glob
import shutil 

def beam_exists(bnum, out_dir):
    bdir = "%s/beam%04d" %(out_dir, bnum)
    return os.path.exists(bdir)


def organize_results(work_dir):
    """
    Put *out + *err files into a directory

    Put *dat + *fft + *inf files into a directory
    """
    dm_dir = "%s/dedisperse" %(work_dir)
    out_dir = "%s/output_files" %(work_dir)

    # Collect all errs and outs
    ostr_list = ["out", "err"]
    for ostr in ostr_list:
        ofiles = glob("%s/*%s" %(work_dir, ostr))
        if len(ofiles):
            # If files exist and out_dir doesnt, make it
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            for ofile in ofiles:
                shutil.move(ofile, out_dir)
        else: pass

    # Collect all infs, dats, and ffts
    dmstr_list = ["inf", "dat", "fft"]
    for dmstr in dmstr_list:
        dmfiles = glob("%s/*%s" %(work_dir, dmstr))
        if len(dmfiles):
            # If files exist and dm_dir doesnt, make it
            if not os.path.exists(dm_dir):
                os.makedirs(dm_dir)
            for dmfile in dmfiles:
                shutil.move(dmfile, dm_dir)
        else: pass

    return


def copy_beam_results(bnum, tmp_dir, out_dir):
    """
    Remove fits files and copy results 
    """
    # WORK AND RESULTS DIRECTORIES
    work_dir  = "%s/beam%04d" %(tmp_dir, bnum)
    res_dir = "%s/beam%04d" %(out_dir, bnum)
    
    # Check for and copy directories
    sub_dirs = ["cands_presto", "dedisperse", "output_files",
                "rfi_products", "single_pulse"]
    for sub_dir in sub_dirs:
        sub_work = "%s/%s" %(work_dir, sub_dir)
        sub_res  = "%s/%s" %(res_dir, sub_dir)
        if not os.path.exists(sub_res):
            if os.path.exists(sub_work):
                shutil.copytree(sub_work, sub_res)
            else: pass
        else: pass

    fitslist = glob("%s/beam*fits" %(work_dir))
    if len(fitslist):
        for fitsfile in fitslist:
            os.remove(fitsfile)
    else: pass
    
    return 


####################
##     MAIN       ##
####################


if __name__ == "__main__":
    gname = 'pulsars'
    search_dir = "/hercules/results/rwharton/fastvis_gc/proc/%s/search" %(gname)
    top_dir = "/hercules/results/rwharton/fastvis_gc/proc/%s" %(gname)
    bname = "mjd57519"  # Basename of FITS files

    start = int(sys.argv[1])
    stop  = int(sys.argv[2])

    print("START = %d" %start)
    print("STOP  = %d" %stop)

    beam_list = range(start, stop)

    plist = range(1,8)

    for bnum in beam_list:
        # Check if beam has already been processed 
        # (this sometimes happens if job fails / restarts)
        if beam_exists(bnum, search_dir):
            print("Beam %04d results exist!" %(bnum))
            pass
        else: 
            print("Beam %04d not found!" %(bnum))
            continue
       
        # Basename + paths in tmp space
        basename = "beam%04d" %(bnum)

        bdir = "%s/beam%04d" %(search_dir, bnum)
       
        # Try Set-up + Searching
        try:
            organize_results(bdir)
            
        # If beam proc fails, continue to next one
        except:
            print("Something failed on %s" %(basename))
            continue
