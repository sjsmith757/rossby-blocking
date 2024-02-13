# Generic methods for writing to disk

import os
import h5py
from typing import Optional
import warnings


def initialize_save_snapshots(self, path):
    """Initializes class variables for saving snapshots.
    Sets the path and creates directory if needed.
    Parameters
    ----------
    path:  string (required)
                Location to save model outputs.
    """

    self.fno = path

    if (not os.path.isdir(self.fno)) & self.save_to_disk:
        os.makedirs(self.fno)
        os.makedirs(self.fno + "/snapshots/")


def save_parameters(self, fields=["t", "A", "F"]):
    """Saves parameters used for the model simulation"""

    if self.save_to_disk:
        fno = self.fno + "/parameters"
        file_exist(fno, overwrite=self.overwrite, stem=".nc" if self.use_xr else ".h5")
        if self.use_xr:
            xr_writer(self, fno, snaps=False)
        else:
            h5writer(self, fno, fields)

    else:
        pass


def file_exist(fno, overwrite=True, stem=".h5"):
    """Check whether file exists.
    Parameters
    ----------
    overwrite:  string (optional)
                    If True, then overwrite extant files.
    """

    if os.path.exists(fno + stem):
        if overwrite:
            os.remove(fno + stem)
        else:
            raise IOError(f"File exists: {fno}")


def h5writer(model, fno: str, fields: list[str], dtypes: Optional[list[str]] = None):
    if dtypes is None:
        dtypes = [None] * len(fields)
    with h5py.File(fno + ".h5", "w") as h5file:
        for field, dtype in zip(fields, dtypes):
            try:
                data = getattr(model, field.split("/")[-1])
            except AttributeError:
                raise ValueError(f"{field} not found in model attributes")
            h5file.create_dataset(field, data=data, dtype=dtype)


def to_dataset(self, snaps: bool = True):
    try:
        import xarray as xr
    except ImportError:
        warnings.warn(
            "cannot convert to dataset without xarray installed", UserWarning, 2
        )
        return

    if snaps:
        var_list = [
            xr.DataArray(
                getattr(self, v)[None],
                dims=["t", "x"],
                coords=[("t", [self.t]), ("x", self.x)],
                name=v,
            )
            for v in ["A", "F", "S", "C"]
        ]
    else:
        var_list = [xr.Coordinates({"x": self.x})]

    var_list.append(xr.Coordinates({"k": self.k}))
    vars = ["nx", "Lx", "dt", "tmax", "tmin", "tau", "Smax", "D", "alpha", "beta"]
    var_list.extend([xr.DataArray(getattr(self, v), name=v) for v in vars])

    ds = xr.merge(var_list)
    params = [
        "printcadence",
        "loglevel",
        "inject",
        "verbose",
        "save_to_disk",
        "overwrite",
        "tsnaps",
        "path",
    ]
    attrs = {p: getattr(self, p) for p in params}
    ds.attrs = {k: int(v) if isinstance(v, bool) else v for k, v in attrs.items()}
    return ds


def xr_writer(model, fno: str, snaps: bool):
    ds = model.to_dataset(snaps=snaps)
    if ds is not None:
        ds.to_netcdf(fno + ".nc")
    else:
        warnings.warn(
            "Output was not written to disk, xarray engine was selected"
            " but is not installed.",
            UserWarning,
            2,
        )


def save_setup(self):
    """Save set up of model simulations."""

    if self.save_to_disk:
        fno = self.fno + "/setup"
        file_exist(fno, overwrite=self.overwrite, stem=".nc" if self.use_xr else ".h5")
        if self.use_xr:
            xr_writer(self, fno, snaps=False)
        else:
            h5writer(
                self, fno, ["grid/nx", "grid/x", "grid/k"], ["int", "float", "float"]
            )

    return


def save_snapshots(self, fields=["t", "A", "F"]):
    """Save snapshots of model simulations.
    Parameters
    ----------
    fields:  list of strings (optional)
                The fields to save. Default is time ('t'),
                wave activity ('A'), and wave-activity flux ('F').
    """

    if self.t >= self.snapstart:
        if (not (self.tc % self.tsnaps)) & (self.save_to_disk):
            fno = self.fno + f"/snapshots/{self.t:015.0f}"
            file_exist(fno, stem=".nc" if self.use_xr else ".h5")
            if self.use_xr:
                xr_writer(self, fno, snaps=True)
            else:
                h5writer(self, fno, fields)


def join_snapshots(self, odir=None):
    if self.use_xr:
        try:
            import xarray as xr
        except ImportError:
            warnings.warn(
                "cannot join dataset without xarray installed", UserWarning, 2
            )
            return

        fno = self.fno + "/snapshots/"
        files = [fno + f for f in list(os.walk(fno))[0][2]]
        ds = xr.open_mfdataset(files)
        if odir is None:
            odir = self.fno
        fout = (
            f"{odir}/{self.path.rstrip('/')}.{files[0].split('/')[-1][:-3]}_"
            f"{files[-1].split('/')[-1][:-3]}.nc"
        )
        ds.to_netcdf(fout)
