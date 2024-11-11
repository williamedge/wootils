import xarray as xr


# @xr.register_dataset_accessor("floatds")
# class dsxrwrap():
#     def __init__(self, ds):
#         self._obj = ds
        
#     @property
#     def ds(self):
#         return self._obj

#     @property
#     def dims(self):
#         return self.ds.dims

#     @property
#     def coords(self):
#         return self.ds.coords
    
#     @property
#     def attrs(self):
#         return self.ds.attrs
    
#     @property
#     def variables(self):
#         return self.ds.variables
    
#     @property
#     def other_dims(self):
#         return [dim for dim in self.ds.dims if (not dim.lower()=='time')]

#     def __repr__(self):
#         return self.ds.__repr__()


# @xr.register_dataset_accessor("xrap")
# @xr.register_dataarray_accessor("xrap")
class xrwrap:
    def __init__(self, xarray_obj):
        if not isinstance(xarray_obj, (xr.Dataset, xr.DataArray)):
            raise TypeError("Expected xarray.Dataset or xarray.DataArray")
        self._obj = xarray_obj

    @property
    def dims(self):
        return self._obj.dims

    @property
    def coords(self):
        return self._obj.coords

    @property
    def attrs(self):
        return self._obj.attrs

    def apply_func(self, func, *args, **kwargs):
        """
        Apply a function to the underlying xarray object.
        """
        return func(self._obj, *args, **kwargs)

    def __repr__(self):
        return f"<xrwrap: {self._obj.__repr__()}>"