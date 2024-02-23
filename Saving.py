# Generic methods for writing to disk

from __future__ import annotations
import os
import h5py
from typing import List, Optional, Tuple, Union, Literal, TYPE_CHECKING
import warnings
import logging
import numpy as np

if TYPE_CHECKING:
    from AtmosphericBlocking import Model

    try:
        import xarray as xr
    except ImportError:
        pass

DtypesList = Optional[List[Optional[str]]]
LogLevelType = Union[
    int, Literal["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
]


class IOInterface(object):
    def __init__(
        self,
        logfile: Optional[str] = "model.out",
        printcadence: int = 1000,
        loglevel: LogLevelType = "INFO",
        save_to_disk: bool = True,
        overwrite: bool = True,
        tsave_snapshots: int = 50,
        tsave_start: float = 0.0,
        verbose: Optional[bool] = None,
        path: str = "output/",
        io_backend: Literal["h5", "xr"] = "h5",
    ):
        """
        primitive wrapper for outputting data from an `AtmosphericBlocking.Model`

        Parameters
        ----------
        logfile : str, optional
            the path to the log file to create, by default "model.out". Setting the
            logfile to none will output the log information to sys.stdout
        printcadence : int, optional
            the frequency to log the model's status, by default 1000
        loglevel : int or ["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                   optional
            the logging level to utilize, by default "INFO"
        save_to_disk : bool, optional
            whether to save the results of a model run, by default True
        overwrite : bool, optional
            whether to overwrite previous cases of the same name, by default True
        tsave_snapshots : int, optional
            how often to save snapshots of the model's internal state, by default 50
        tsave_start : float, optional
            when to begin saving snapshots, by default 0.0
        verbose : bool, optional
            deprecated parameter. loglevel should be used instead, by default None
        path : str, optional
            the root directory for the model experiment, by default "output/"
        io_backend : ["h5", "xr"], optional
            the backedn to use for writing data, by default "h5"
        """

        if verbose is not None:
            warnings.warn(
                "the verbose argument is deprecated and will soon be removed",
                DeprecationWarning,
            )
        else:
            verbose = False
        self._verbose = verbose

        self.printcadence = printcadence
        self.loglevel = loglevel
        self.logfile = logfile
        self.save_to_disk = save_to_disk
        self.overwrite = overwrite
        self.tsnaps = tsave_snapshots
        self.snapstart = tsave_start
        self.path = path

        self.use_xr: bool = False
        if io_backend == "xr":
            try:
                import xarray  # noqa: F401

                self.use_xr = True
            except ImportError:
                warnings.warn(
                    "xarray backend was selected, but xarray is not installed."
                    " Defaulting to h5py backend.",
                    UserWarning,
                    2,
                )

        # initializations
        self.t: float = 0.0
        self.tc: int = 0
        self._initialize_logger()
        self.initialize_save_snapshots(self.path)

    def initialize_save_snapshots(self, path: str):
        """
        Initializes class variables for saving snapshots. Sets the path and creates
        directory if needed.

        Parameters
        ----------
        path:  string or os.PathLike (required)
                    Location to save model outputs.
        """

        if (not os.path.isdir(path)) & self.save_to_disk:
            os.makedirs(path)
            os.makedirs(path + "/snapshots/")

    def save_parameters(self, fields: List[str] = []):
        """
        Saves parameters used for the model simulation

        Parameters
        ----------

        fields: list of str, optional
                the fields to save

        """

        if self.save_to_disk:
            fno = self.path + "/parameters"
            file_exist(
                fno, overwrite=self.overwrite, stem=".nc" if self.use_xr else ".h5"
            )
            if self.use_xr:
                xr_writer(self, fno, fields=[])
            else:
                fields += [
                    "printcadence",
                    "loglevel",
                    "save_to_disk",
                    "overwrite",
                    "tsnaps",
                    "path",
                ]
                h5writer(self, fno, fields)

        else:
            pass

    def to_dataset(
        self,
        coords: List[Tuple[str, np.ndarray]] = [],
        dvars: List[str] = [],
        params: List[str] = [],
    ) -> xr.Dataset:
        """
        If xarray is installed, convert the model internals to a dataset for output as a
        netcdf file

        Parameters
        ----------
        coords : list of (str, :py:class:`np.ndarray`), optional
            a list of tuple containing the coordinate names and coordinate values
        dvars: list of str, optional
            a list of the dimensionalized variables to include
        params:
            a list of metadata variables to include

        Returns
        -------
        xr.Dataset
            a Dataset containing the model internals
        """
        try:
            import xarray as xr
        except ImportError:
            raise ValueError("cannot convert to dataset without xarray installed")

        var_list: List[Union[xr.DataArray, xr.Coordinates]] = []
        for v in dvars:
            if v not in [c for c, _ in coords]:
                arr = np.array(getattr(self, v))
                if arr.squeeze().ndim == 1:
                    da = xr.DataArray(
                        arr[None],
                        dims=[coords[0][0], coords[1][0]],
                        coords=coords[:2],
                        name=v,
                    )
                else:
                    da = xr.DataArray(arr, name=v)
                var_list.append(da)

        ds = xr.merge(var_list)

        for cn, cv in coords:
            if cn not in ds:
                ds[cn] = xr.Coordinates({cn: cv})

        params.extend(
            [
                "printcadence",
                "loglevel",
                "save_to_disk",
                "overwrite",
                "tsnaps",
                "path",
            ]
        )
        attrs = {p: getattr(self, p) for p in params}
        ds.attrs = {k: int(v) if isinstance(v, bool) else v for k, v in attrs.items()}
        return ds

    def save_setup(self, fields: List[str], dtypes: DtypesList = None):
        """
        Save set up of model simulations.

        Parameters
        ---------

        fields : list of str, optional
                a list of data variables to save
        dtypes : list of str or None, optional
                a list of data types for the fields. If None, assumed from the data.

        """

        if self.save_to_disk:
            fno = self.path + "/setup"
            file_exist(
                fno, overwrite=self.overwrite, stem=".nc" if self.use_xr else ".h5"
            )
            if self.use_xr:
                xr_writer(self, fno, fields=[])
            else:
                h5writer(self, fno, fields, dtypes=dtypes)

        return

    def save_snapshots(self, fields: List[str] = []):
        """
        Save snapshots of model simulations.

        Parameters
        ----------
        fields:  list of strings (optional)
                    The fields to save.
        """

        if self.t >= self.snapstart:
            if (not (self.tc % self.tsnaps)) & (self.save_to_disk):
                fno = self.path + f"/snapshots/{self.t:015.0f}"
                file_exist(fno, stem=".nc" if self.use_xr else ".h5")
                if self.use_xr:
                    xr_writer(self, fno, fields=fields)
                else:
                    h5writer(self, fno, fields)

    def join_snapshots(self, odir: Optional[str] = None):
        """
        combine individual temporal snapshots from the model into a single output file. This
        is currently only supported if using the xarray backend (`io_backend="xr"`)

        Parameters
        ----------
        odir : str, optional
            the output directory for the combined file, by default the experiment's root
            directory (`path`)
        """
        if self.use_xr:
            try:
                import xarray as xr
            except ImportError:
                warnings.warn(
                    "cannot join dataset without xarray installed", UserWarning, 2
                )
                return

            fno = self.path + "/snapshots/"
            files = [fno + f for f in list(os.walk(fno))[0][2]]
            ds = xr.open_mfdataset(files)
            if odir is None:
                odir = self.path
            fout = (
                f"{odir}/{self.path.rstrip('/')}.{files[0].split('/')[-1][:-3]}_"
                f"{files[-1].split('/')[-1][:-3]}.nc"
            )
            ds.to_netcdf(fout)

    def _print_status(self) -> None:
        """Output some basic stats."""
        if (self.tc % self.printcadence) == 0:
            self.logger.info(f"Step: {self.tc:4d}, Time: {self.t:3.2e}")
        pass

    # logger
    def _initialize_logger(self) -> None:
        """initialize the logger"""

        self.logger = logging.getLogger(__name__)

        fhandler: logging.Handler
        if self.logfile:
            fhandler = logging.FileHandler(filename=self.logfile, mode="w")
        else:
            fhandler = logging.StreamHandler()

        formatter = logging.Formatter("%(levelname)s: %(message)s")

        fhandler.setFormatter(formatter)

        if not self.logger.handlers:
            self.logger.addHandler(fhandler)

        self.logger.setLevel(self.loglevel)

        # this prevents the logger to propagate into the ipython notebook log
        self.logger.propagate = False

        self.logger.info(" Logger initialized")


def file_exist(fno: str, overwrite: bool = True, stem: str = ".h5"):
    """Check whether file exists.
    Parameters
    ----------
    overwrite:  boolean (optional)
                    If True, then overwrite extant files.
    stem:       string, optional
                    The file extension to use
    """

    if os.path.exists(fno + stem):
        if overwrite:
            os.remove(fno + stem)
        else:
            raise IOError(f"File exists: {fno}")


def h5writer(
    model: IOInterface, fno: str, fields: List[str], dtypes: DtypesList = None
):
    """
    file creator for outputing model internals to am HDF5 file

    Parameters
    ----------
    model : Model
        An instance of an AtmosphericBlocking.Model to output
    fno : str
        the file name to create
    fields : List[str]
        the fields to write to the file
    dtypes : List[str], optional
        the datatypes for the output fields, by default inferred from the data
    """

    if dtypes is None:
        dtypes = [None] * len(fields)
    with h5py.File(fno + ".h5", "w") as h5file:
        for field, dtype in zip(fields, dtypes):
            try:
                data = getattr(model, field.split("/")[-1])
            except AttributeError:
                raise ValueError(f"{field} not found in model attributes")
            h5file.create_dataset(field, data=data, dtype=dtype)


def xr_writer(model: Union[IOInterface, Model], fno: str, fields: List[str]):
    """
    file creater for saving model internals as a netcdf file

    Parameters
    ----------
    model : Model
        an instance of an AtmosphericBlocking.Model
    fno : str
        the name of the output file to create
    snaps : bool
        whether to include the model temporally varying fields
    """
    ds = model.to_dataset(dvars=fields)
    if ds is not None:
        ds.to_netcdf(fno + ".nc")
    else:
        warnings.warn(
            "Output was not written to disk, xarray engine was selected"
            " but is not installed.",
            UserWarning,
            2,
        )
