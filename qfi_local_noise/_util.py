import sys
import re
import pprint
import inspect

import logging
module_logger = logging.getLogger()


import numpy as np
import scipy.special as spa





# General "Namespace" utility which stores values associated with keys. Really
# just a dictionary with .member access instead of ['member'] which is
# cumbersome to type.
class NS:
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)

    def _NS_repr(self, clsname=None):
        if clsname is None:
            clsname = self.__class__.__name__
        try:
            nppr = np.get_printoptions()
            np.set_printoptions(threshold=8, edgeitems=1, suppress=True)
            return clsname + \
                pprint.pformat({k: v
                                for k, v in self.__dict__.items()
                                if not k.startswith('_')},
                               depth=1)
        finally:
            np.set_printoptions(**nppr)

    def __repr__(self):
        return self._NS_repr()


# ------------------------------------------------------------------------------


def abs2(x):
    return x.real**2 + x.imag**2



def multinomial_coefficient(mlist, exact=True):
    #
    # multinomial = (n choose m[0])*(n-m[0] choose m[1])*(n-m[0]-m[1] choose m[2])*...
    #

    # nlist is [ n, n-m[0], n-m[0]-m[1], ..., m[-1] ]
    #          == [ m[0]+...+m[-1], ..., m[-2]+m[-1], m[-1] ]
    nlist = np.cumsum(mlist[::-1])[::-1]

    mcoeff = 1
    for j in range(len(mlist)-1):
        mcoeff = mcoeff * spa.comb(nlist[j], mlist[j], exact=exact)

    return mcoeff




#
# Thanks https://stackoverflow.com/a/42625707/1694896 !!
#
def iter_spmatrix(matrix):
    """ Iterator for iterating the elements in a ``scipy.sparse.*_matrix`` 

    This will always return:
    >>> (row, column, matrix-element)

    Currently this can iterate `coo`, `csc`, `lil` and `csr`, others may easily be added.

    Parameters
    ----------
    matrix : ``scipy.sparse.sp_matrix``
      the sparse matrix to iterate non-zero elements
    """
    import scipy.sparse as sps

    if sps.isspmatrix_coo(matrix):
        for r, c, m in zip(matrix.row, matrix.col, matrix.data):
            yield r, c, m

    elif sps.isspmatrix_csc(matrix):
        for c in range(matrix.shape[1]):
            for ind in range(matrix.indptr[c], matrix.indptr[c+1]):
                yield matrix.indices[ind], c, matrix.data[ind]

    elif sps.isspmatrix_csr(matrix):
        for r in range(matrix.shape[0]):
            for ind in range(matrix.indptr[r], matrix.indptr[r+1]):
                yield r, matrix.indices[ind], matrix.data[ind]

    elif sps.isspmatrix_lil(matrix):
        for r in range(matrix.shape[0]):
            for c, d in zip(matrix.rows[r], matrix.data[r]):
                yield r, c, d

    else:
        raise NotImplementedError("The iterator for this sparse matrix has not been implemented")







# ------------------------------------------------------------------------------



# internal helper to validate inputs to functions & methods

# *** for _validate() ***
#
# the symbol that indicates a local expression whose value we should display
_showvalue_symbol = '$'
#
# regex that picks up a marked local expression e.g. "$self.localdim".  The
# regex should look like r"\$(expression-chars-including-dot-etc.)+"
_rx_varname_showvalue = re.compile('\\'+_showvalue_symbol+r'(?P<varname>[a-zA-Z0-9._]+)')

def _validate(s):
    """
    Internal tool to validate inputs.  The argument `s` is a string containing
    an "annotated" Python expression.  The expression is evaluated; if a
    non-True value is obtained then a `ValueError` is raised.  The annotations
    are used to display values of local variables that might help debug what
    input was incorrect.

    The annotations are of the form "$variable_name" or
    "$variable_name.field_name" (instead of "variable_name" or
    "variable_name.field_name").  Example::

        _validate("$a >= $b") # assuming we have local variables `a` and `b`

    An expression annotated with a '$' sign can contain standard variable names
    (a-z, A-Z, underscores and a dot (for field names)).
    """
    frame = inspect.currentframe()
    caller_frame = frame.f_back

    def eval_in_caller_scope(x):
        #return eval(x, dict(caller_frame.f_globals), dict(caller_frame.f_locals))
        #
        # pretend locals are globals to work around python bug
        # https://bugs.python.org/issue36300, and make code like
        # _validate(r""" np.all([ E.shape == [self.a, self.b] for E in Eks]) """)
        return eval(x, dict(caller_frame.f_globals, **caller_frame.f_locals), {})

    s_expr = re.sub(r'\s{2,}', '  ', s.replace(_showvalue_symbol, '')).strip()
    what_value = eval_in_caller_scope(s_expr)

    if 'validate_logger' in caller_frame.f_locals:
        logger = caller_frame.f_locals['validate_logger']
    elif 'validate_logger' in caller_frame.f_globals:
        logger = caller_frame.f_globals['validate_logger']
    elif 'logger' in caller_frame.f_locals:
        logger = caller_frame.f_locals['logger']
    elif 'logger' in caller_frame.f_globals:
        logger = caller_frame.f_globals['logger']
    else:
        logger = module_logger
        
    logger.debug("Check %s → %s", s_expr, what_value)

    if not what_value:
        # failure--raise error and show variable values.

        #merged_caller_scope = dict(caller_frame.f_globals, **caller_frame.f_locals)
        #logger = merged_caller_scope.get('logger', module_logger)

        the_vars = dict()
        for m in _rx_varname_showvalue.finditer(s):
            varname = m.group('varname')
            the_vars[varname] = eval_in_caller_scope(varname)
            #merged_caller_scope.get(varname, '<UNKNOWN>')

        #logger.debug("the_vars = %r", the_vars)

        with_vars = ""
        if the_vars:
            with_vars = "  [with\n{}\n]".format(
                ",\n".join([f'    {xn}={xv!r}' for xn,xv in the_vars.items()])
            )

        #logger.debug("with_vars = %r", with_vars)

        logger.error("Input condition ‘{s}’ violated".format(s=s_expr)
                     + with_vars)

        raise ValueError("Invalid input, condition ‘{s}’ violated".format(s=s_expr)
                         + with_vars)





# ------------------------------------------------------------------------------

# tool for multiprocessing.
#
# see https://github.com/ipython/ipython/issues/11049#issue-306086846
def multiprocessing_streams_init_pool():
    sys.stdout.write(".\b") # need to send *something* over stdout to use ipython's display()
    sys.stdout.flush()
