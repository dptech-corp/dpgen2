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
    argo_range,
)
from dflow.python import(
    PythonOPTemplate,
    OP,
    OPIO,
    OPIOSign,
    Artifact,
    Slices,
)

from typing import Set, List
from pathlib import Path


def steps_train(
        name : str,
        make_train_op : OP,
        run_train_op : OP,
):
    train_steps = Steps(name=name,
                        inputs=Inputs(
                            parameters={
                                "numb_models": InputParameter(type=int),
                                "template_script" : InputParameter(),
                            },
                            artifacts={
                                "init_models" : InputArtifact(),
                                "init_data" : InputArtifact(),
                                "iter_data" : InputArtifact(),
                            },
                        ),
                        outputs=Outputs(
                            artifacts={
                                "scripts": OutputArtifact(),
                                "models": OutputArtifact(),
                                "logs": OutputArtifact(),
                                "lcurves": OutputArtifact(),
                            }),
                        )

    make_train = Step('make-train',
                      template=PythonOPTemplate(
                          make_train_op,
                          image="dflow:v1.0",
                          output_artifact_archive={
                              "train_scripts": None
                          },
                          # python_packages = "..//dpgen2",
                      ),
                      parameters={
                          "numb_models": train_steps.inputs.parameters['numb_models'],
                          "template_script": train_steps.inputs.parameters['template_script'],
                      },
                      artifacts={
                      },
                      )
    train_steps.add(make_train)

    run_train = Step('run-train',
                     template=PythonOPTemplate(
                         run_train_op,
                         image="dflow:v1.0",
                         slices = Slices(
                             "{{item}}",
                             input_parameter = ["task_subdir"],
                             input_artifact = ["train_script", "init_model"],
                             output_artifact = ["model", "lcurve", "log", "script"],
                         ),
                         # python_packages = "..//dpgen2",
                     ),
                     parameters={
                         "task_subdir" : make_train.outputs.parameters["task_subdirs"],
                     },
                     artifacts={
                         'train_script' : make_train.outputs.artifacts['train_scripts'],
                         "init_model" : train_steps.inputs.artifacts['init_models'],
                         "init_data": train_steps.inputs.artifacts['init_data'],
                         "iter_data": train_steps.inputs.artifacts['iter_data'],
                     },
                     with_param=argo_range(train_steps.inputs.parameters["numb_models"]),
                     )
    train_steps.add(run_train)

    train_steps.outputs.artifacts["scripts"]._from = run_train.outputs.artifacts["script"]
    train_steps.outputs.artifacts["models"]._from = run_train.outputs.artifacts["model"]
    train_steps.outputs.artifacts["logs"]._from = run_train.outputs.artifacts["log"]
    train_steps.outputs.artifacts["lcurves"]._from = run_train.outputs.artifacts["lcurve"]

    return train_steps


