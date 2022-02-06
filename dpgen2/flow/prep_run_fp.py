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
    argo_len,
    argo_sequence,
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


def prep_run_fp(
        name : str,
        prep_op : OP,
        run_op : OP,
        upload_python_package : str = None,
):
    prep_run_steps = Steps(
        name=name,
        inputs=Inputs(
            parameters={
                "inputs": InputParameter(),
                "fp_config" : InputParameter(),
            },
            artifacts={
                "confs" : InputArtifact()
            },
        ),
        outputs=Outputs(
            parameters={
                "task_names": OutputParameter(),
            },
            artifacts={
                "logs": OutputArtifact(),
                "labeled_data": OutputArtifact(),
            }),
    )

    prep_fp = Step(
        'prep-fp',
        template=PythonOPTemplate(
            prep_op,
            image="dflow:v1.0",
            output_artifact_archive={
                "task_paths": None
            },
            python_packages = upload_python_package,
        ),
        parameters={
            "inputs": prep_run_steps.inputs.parameters['inputs'],
        },
        artifacts={
            "confs" : prep_run_steps.inputs.artifacts['confs'],
        },
    )
    prep_run_steps.add(prep_fp)

    run_fp = Step(
        'run-fp',
        template=PythonOPTemplate(
            run_op,
            image="dflow:v1.0",
            slices = Slices(
                "{{item}}",
                input_parameter = ["task_name"],
                input_artifact = ["task_path"],
                output_artifact = ["log", "labeled_data"],
            ),
            python_packages = upload_python_package,
        ),
        parameters={
            "task_name" : prep_fp.outputs.parameters["task_names"],
            "config" : prep_run_steps.inputs.parameters["fp_config"],
        },
        artifacts={
            'task_path' : prep_fp.outputs.artifacts['task_paths'],
        },
        with_param=argo_range(argo_len(prep_fp.outputs.parameters["task_names"])),
    )
    prep_run_steps.add(run_fp)

    prep_run_steps.outputs.parameters["task_names"].value_from_parameter = prep_fp.outputs.parameters["task_names"]
    prep_run_steps.outputs.artifacts["logs"]._from = run_fp.outputs.artifacts["log"]
    prep_run_steps.outputs.artifacts["labeled_data"]._from = run_fp.outputs.artifacts["labeled_data"]

    return prep_run_steps


