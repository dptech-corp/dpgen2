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


def prep_run_lmp(
        name : str,
        prep_op : OP,
        run_op : OP,
):
    prep_run_steps = Steps(
        name=name,
        inputs=Inputs(
            parameters={
                "lmp_task_grp": InputParameter(type=int),
                "lmp_config" : InputParameter()
            },
            artifacts={
                "models" : InputArtifact()
            },
        ),
        outputs=Outputs(
            parameters={
                "task_names": OutputParameter(),
            },
            artifacts={
                "logs": OutputArtifact(),
                "trajs": OutputArtifact(),
                "model_devis": OutputArtifact(),
            }),
    )

    prep_lmp = Step(
        'prep-lmp',
        template=PythonOPTemplate(
            prep_op,
            image="dflow:v1.0",
            output_artifact_archive={
                "task_paths": None
            },
            python_packages = "..//dpgen2",
        ),
        parameters={
            "lmp_task_grp": prep_run_steps.inputs.parameters['lmp_task_grp'],
        },
        artifacts={
        },
    )
    prep_run_steps.add(prep_lmp)

    run_lmp = Step(
        'run-lmp',
        template=PythonOPTemplate(
            run_op,
            image="dflow:v1.0",
            slices = Slices(
                "{{item}}",
                input_parameter = ["task_name"],
                input_artifact = ["task_path"],
                output_artifact = ["log", "traj", "model_devi"],
            ),
            python_packages = "..//dpgen2",
        ),
        parameters={
            "task_name" : prep_lmp.outputs.parameters["task_names"],
            "config" : prep_run_steps.inputs.parameters["lmp_config"],
        },
        artifacts={
            'task_path' : prep_lmp.outputs.artifacts['task_paths'],
            "models" : prep_run_steps.inputs.artifacts['models'],
        },
        # with_sequence=argo_sequence(argo_len(prep_lmp.outputs.parameters["task_names"])),
        with_param=argo_range(argo_len(prep_lmp.outputs.parameters["task_names"])),
    )
    prep_run_steps.add(run_lmp)

    prep_run_steps.outputs.parameters["task_names"].value_from_parameter = prep_lmp.outputs.parameters["task_names"]
    prep_run_steps.outputs.artifacts["logs"]._from = run_lmp.outputs.artifacts["log"]
    prep_run_steps.outputs.artifacts["trajs"]._from = run_lmp.outputs.artifacts["traj"]
    prep_run_steps.outputs.artifacts["model_devis"]._from = run_lmp.outputs.artifacts["model_devi"]

    return prep_run_steps


