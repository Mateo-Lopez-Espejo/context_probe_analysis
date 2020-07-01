import numpy as np
import pandas as pd
'''
a growing collection of general usefull functions???

'''

def establish_variables(self, x=None, y=None, hue=None, data=None,
                        orient=None, order=None, hue_order=None,
                        units=None):
    """Convert input specification into a common representation.
    Taken from seaborn codebase"""

    # See if we need to get variables from `data`
    if data is not None:
        x = data.get(x, x)
        y = data.get(y, y)
        hue = data.get(hue, hue)
        units = data.get(units, units)

    # Validate the inputs
    for var in [x, y, hue, units]:
        if isinstance(var, str):
            err = "Could not interpret input '{}'".format(var)
            raise ValueError(err)

    # Figure out the plotting orientation
    orient = self.infer_orient(x, y, orient)

    # Option 2b:
    # We are grouping the data values by another variable
    # ---------------------------------------------------

    # Determine which role each variable will play
    if orient == "v":
        vals, groups = y, x
    else:
        vals, groups = x, y

    # Get the categorical axis label
    if hasattr(groups, "name"):
        group_label = groups.name
    else:
        group_label = None

    # Get the order on the categorical axis
    group_names = categorical_order(groups, order)

    # Group the numeric data
    plot_data, value_label = self._group_longform(vals, groups,
                                                  group_names)

    # Now handle the hue levels for nested ordering
    if hue is None:
        plot_hues = None
        hue_title = None
        hue_names = None
    else:

        # Get the order of the hue levels
        hue_names = categorical_order(hue, hue_order)

        # Group the hue data
        plot_hues, hue_title = self._group_longform(hue, groups,
                                                    group_names)

    # Now handle the units for nested observations
    if units is None:
        plot_units = None
    else:
        plot_units, _ = self._group_longform(units, groups,
                                             group_names)

    # Assign object attributes
    # ------------------------
    self.orient = orient
    self.plot_data = plot_data
    self.group_label = group_label
    self.value_label = value_label
    self.group_names = group_names
    self.plot_hues = plot_hues
    self.hue_title = hue_title
    self.hue_names = hue_names
    self.plot_units = plot_units


def categorical_order(values, order=None):
    """Return a list of unique data values.

    Determine an ordered list of levels in ``values``.

    Parameters
    ----------
    values : list, array, Categorical, or Series
        Vector of "categorical" values
    order : list-like, optional
        Desired order of category levels to override the order determined
        from the ``values`` object.

    Returns
    -------
    order : list
        Ordered list of category levels not including null values.

    """
    if order is None:
        if hasattr(values, "categories"):
            order = values.categories
        else:
            try:
                order = values.cat.categories
            except (TypeError, AttributeError):
                try:
                    order = values.unique()
                except AttributeError:
                    order = pd.unique(values)
                try:
                    np.asarray(values).astype(np.float)
                    order = np.sort(order)
                except (ValueError, TypeError):
                    order = order
        order = filter(pd.notnull, order)
    return list(order)