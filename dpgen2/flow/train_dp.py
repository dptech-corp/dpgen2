from dflow import (
    InputParameter,
    OutputParameter,
    Inputs,
    InputArtifact,
    Outputs,
    OutputArtifact,
    Workflow,
    Step,
    Steps,
    upload_artifact,
    download_artifact,
    S3Artifact,
    argo_range
)
from dflow.python import(
    PythonOPTemplate,
    OP,
    OPIO,
    OPIOSign,
    Artifact
)

from typing import Set, List
from pathlib import Path

class CollectResult(OP):
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            'numb_models': int,
            'outcar' : Artifact(Path)
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            'outcar' : Artifact(List[Path]),
        })

    @OP.exec_sign_check
    def execute(
            self,
            op_in : OPIO,
    ) -> OPIO:
        numb_vasp = op_in['numb_vasp']
        olist = []
        for ii in range(numb_vasp):
            ofile = op_in['outcar'] / f'task.{ii:04d}' / 'OUTCAR'
            olist.append(ofile)
        return OPIO({
            'outcar': olist
        })


def steps_train(
        numb_models : int,
        template_script : Artifact(Path),
        init_model : Artifact(List[Path]),
        init_data : Artifact(Set[Path]),
        iter_data : Artifact(Set[Path]),
        make_train_op : OP,
        run_train_op : OP,
        prefix = '',
        suffix = '',
):
    train_steps = Steps(name="vasp-steps",
                        inputs=Inputs(
                            parameters={
                                "numb_models": InputParameter(type=int),
                                "template_script" : InputParameter(),
                                "init_model" : InputArtifact(),
                                "init_data" : InputArtifact(),
                                "iter_data" : InputArtifact(),
                            })
                        )

    make_train = Step(prefix + 'make-train' + suffix,
                      template=PythonOPTemplate(
                          make_train_op,
                          image="dflow:v1.0",
                          output_artifact_archive={
                              "train_scripts": None
                          }),
                      parameters={
                          "numb_models": train_steps.inputs.parameters['numb_vasp'],
                          "template_script": train_steps.inputs.parameters['template_script'],
                      },
                      artifacts={
                          "init_data": train_steps.inputs.artifacts['init_data'],
                          "iter_data": train_steps.inputs.artifacts['iter_data'],
                      },
                      )
    train_steps.add(make_train)
    
    run_train = Step(prefix + 'run-train' + suffix,
                     template=PythonOPTemplate(
                         RunVasp,
                         image="dflow:v1.0",
                         input_artifact_slices={
                             "train_scripts": "{{item}}",
                             "init_model": "{{item}}"
                         },
                         output_artifact_save={
                             "model": artifact,
                             "lcurve": artifact,
                             "log": artifact,
                         },
                         output_artifact_archive={
                             "model": None,
                             "lcurve": None,
                             "log": None,
                         }),
                     artifacts={
                         'train_script' : make_train.outputs.artifacts['train_scripts'],
                         "init_model" : train_steps.inputs.artifacts['init_model'],
                         "init_data": train_steps.inputs.artifacts['init_data'],
                         "iter_data": train_steps.inputs.artifacts['iter_data'],
                     },
                     with_param=argo_range(train_steps.inputs.parameters["numb_models"])
                     )
    train_steps.add(run_train)

    return train_steps


