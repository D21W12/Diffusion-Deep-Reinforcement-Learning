# Import Order

Guide lines for ordering import statements:

```python
import a_standard
import b_standard

import a_third_party
import b_third_party

from a_soc import f
from a_soc import g
from b_soc import d
```

> ### Example
>
> ```python
> import math
> 
> import torch
> import torch.nn as nn
> 
> from . import DQN
> ```