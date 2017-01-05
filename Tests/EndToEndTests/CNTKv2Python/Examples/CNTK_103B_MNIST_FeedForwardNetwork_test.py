# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import re

abs_path = os.path.dirname(os.path.abspath(__file__))
notebook = os.path.join(abs_path, "..", "..", "..", "..", "Tutorials", "CNTK_103B_MNIST_FeedForwardNetwork.ipynb")

def test_cntk_103_mnist_feedforwardnetwork_noErrors(nb):
    errors = [output for cell in nb.cells if 'outputs' in cell
              for output in cell['outputs'] if output.output_type == "error"]
    assert errors == []

expectedEvalError = '1.79'

def test_cntk_103_mnist_feedforwardnetwork_evalCorrect(nb):
    testCell = [cell for cell in nb.cells
                if cell.cell_type == 'code' and cell.source.find("print(\"Average test error:") != -1]
    testCell = [cell for cell in nb.cells
                if cell.cell_type == 'code' and re.search('trainer\.test_minibatch', cell.source)]
    assert len(testCell) == 1
    m = re.match(r"Average test error: (?P<actualEvalError>\d+\.\d+)%\r?$", testCell[0].outputs[0]['text'])
    assert m.group('actualEvalError') == expectedEvalError
