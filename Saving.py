# Generic methods for writing to disk

import os
import h5py

def initialize_save_snapshots(self,path):

    """ Initializes class variables for saving snapshots.
        Sets the path and creates directory if needed.
        Parameters
        ----------
        path:  string (required)
                    Location to save model outputs.
    """

    self.fno = path

    if (not os.path.isdir(self.fno)) & self.save_to_disk:
        os.makedirs(self.fno)
        os.makedirs(self.fno+"/snapshots/")

def file_exist(fno,overwrite=True):

    """ Check whether file exists.
        Parameters
        ----------
        overwrite:  string (optional)
                        If True, then overwrite extant files.
    """

    if os.path.exists(fno):
        if overwrite:
            os.remove(fno)
        else: raise IOError("File exists: {0}".format(fno))

def save_setup(self,):

    """ Save set up of model simulations.
    """

    if self.save_to_disk:

        fno = self.fno + '/setup.h5'

        file_exist(fno,overwrite=self.overwrite)

        h5file = h5py.File(fno, 'w')
        h5file.create_dataset("grid/nx", data=(self.nx),dtype=int)
        h5file.create_dataset("grid/x", data=(self.x))
        h5file.create_dataset("grid/k", data=self.k)
        h5file.close()

def save_snapshots(self, fields=['t','A','F']):

    """ Save snapshots of model simulations.
        Parameters
        ----------
        fields:  list of strings (optional)
                    The fields to save. Default is time ('t'),
                    wave activity ('A'), and wave-activity flux ('F').
    """

    if ( ( not (self.tc%self.tsnaps) ) & (self.save_to_disk) ):

        fno = self.fno + '/snapshots/{:015.0f}'.format(self.t)+'.h5'

        file_exist(fno)

        h5file = h5py.File(fno, 'w')

        for field in fields:
            if field == 't':
                h5file.create_dataset(field, data=(self.t))
            else:
                h5file.create_dataset(field, data=eval("self."+field))

        h5file.close()
    else:
        pass

def save_diagnostics(self):

    """ Save diagnostics of model simulations.
    """
    fno = self.fno + '/diagnostics.h5'

    file_exist(fno,overwrite=self.overwrite)

    h5file = h5py.File(fno, 'w')

    for key in self.diagnostics.keys():
        h5file.create_dataset(key, data=(self.diagnostics[key]['value']))

    h5file.close()
