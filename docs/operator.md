
# Operators

The operators are building blocks of the workflow. 

DPGEN2 implements the OPs in Python. All OPs are derived from the base class `dflow.OP`. An example `OP` `CollectData` is provided as follows.

```python
from dflow.python import (
    OP,
    OPIO,
    OPIOSign,
    Artifact
)

class CollectData(OP):
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "name" : str,
            "labeled_data" : Artifact(List[Path]),
            "iter_data" : Artifact(Set[Path]),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "iter_data" : Artifact(Set[Path]),
        })

    @OP.exec_sign_check
    def execute(
            self,
            ip : OPIO,
    ) -> OPIO:
        name = ip['name']
        labeled_data = ip['labeled_data']
        iter_data = ip['iter_data']

        ## do works to generate new_iter_data
        ...
        ## done
        
        return OPIO({
            "iter_data" : new_iter_data,
        })
```

The `dflow` requires static type define, i.e. the signatures of an OP, for the input and output variables. The input and output signatures of the `OP` are given by `classmethods` `get_input_sign` and `get_output_sign`. 

The operator is executed by the method `OP.executed`. The inputs and outputs variables are recorded in `dict`s. The keys in the input/output `dict`, and the types of the input/output variables will be checked against their signatures by decorator `OP.exec_sign_check`. If any key or type does not match, an exception will be raised.

The python `OP`s will be wrapped to `dflow` operators (named `Step`) to construct the workflow. An example of wrapping is 
```python
    collect_data = Step(
        name = "collect-data"
        template=PythonOPTemplate(
            CollectData,
            image="dflow:v1.0",
        ),
        parameters={
            "name": foo.inputs.parameters["name"],
        },
        artifacts={
            "iter_data" : foo.inputs.artifacts['iter_data'],
            "labeled_data" : bar.outputs.artifacts['labeled_data'],
        },
    )
```
