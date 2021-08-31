import logging
import unittest

import numpy as np


from qfi_local_noise._util import (
    _validate,
)

global_var = [1, 2, 3]

class TestValidate(unittest.TestCase):
    def test_simple(self):
        local_var = np.ones(shape=(1, 2, 3))
        _validate(r""" list($local_var.shape) == $global_var """)

        _validate(r""" list($local_var .shape) == $global_var """)

    def test_newlines(self):
        local_var = np.ones(shape=(1, 2, 3))
        local_var2 = np.ones(shape=(1, 2, 4))
        _validate(r""" list($local_var.shape)
                       == $global_var """)
        with self.assertRaises(ValueError):
            _validate(r""" list($local_var2 .shape)
                           == $global_var """)

    def test_failure(self):
        local_var = np.ones(shape=(1, 2, 4))

        with self.assertRaises(ValueError):
            _validate(r""" list($local_var.shape) == $global_var """)

        with self.assertRaises(ValueError):
            _validate(r""" list($local_var .shape) == $global_var """)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
