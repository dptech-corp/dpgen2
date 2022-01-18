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
import jsonpickle
from typing import (
    List
)
from pathlib import Path
from dpgen2.exploration.scheduler import ExplorationScheduler
from dpgen2.exploration.report import ExplorationReport
from dpgen2.utils.lmp_task_group import LmpTaskGroup
from dpgen2.utils.conf_selector import ConfSelector
from dpgen2.flow.block import block_cl

class SchedulerWrapper(OP):

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "exploration_scheduler" : ExplorationScheduler,
            "exploration_report": ExplorationReport,
            "trajs": Artifact(List[Path]),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "exploration_scheduler" : ExplorationScheduler,
            "converged" : bool,
            "lmp_task_grp" : LmpTaskGroup,
            "conf_selector" : ConfSelector,
        })

    @OP.exec_sign_check
    def execute(
            self,
            ip : OPIO,
    ) -> OPIO:
        scheduler = ip['exploration_scheduler']
        report = ip['exploration_report']
        trajs = ip['trajs']
        
        conv, lmp_task_grp, selector = scheduler.plan_next_iteration(report, trajs)
        
        return OPIO({
            "exploration_scheduler" : scheduler,
            "converged" : conv,
            "lmp_task_grp" : lmp_task_grp,
            "conf_selector" : selector,
        })

    


def loop (
        name : str,
        block_op : OP,
):
    steps = Steps(
        name = name,
        inputs = Inputs(
            parameters={
                "name_suffix" : InputParameter(),
                "type_map" : InputParameter(),
                "numb_models": InputParameter(type=int),
                "template_script" : InputParameter(),
                "lmp_task_grp" : InputParameter(),
                "conf_selector" : InputParameter(),
                "conf_filters" : InputParameter(),
                "fp_inputs" : InputParameter(),
                "exploration_scheduler" : InputParameter(),
            },
            artifacts={
                "init_models" : InputArtifact(),
                "init_data" : InputArtifact(),
                "iter_data" : InputArtifact(),
            },
        ),
        # outputs=Outputs(
        #     parameters={
        #         "exploration_scheduler": OutputParameter(),
        #     },
        #     artifacts={
        #         "models": OutputArtifact(),
        #         "iter_data" : OutputArtifact(),
        #     },
        # ),
    )

    # suffix = steps.inputs.parameters["name_suffix"].value
    suffix=''

    block_step = Step(
        name + suffix,
        template = block_op,
        parameters={
            "type_map" : steps.inputs.parameters["type_map"],
            "numb_models" : steps.inputs.parameters["numb_models"],
            "template_script" : steps.inputs.parameters["template_script"],
            "lmp_task_grp" : steps.inputs.parameters["lmp_task_grp"],
            "conf_selector" : steps.inputs.parameters["conf_selector"],
            "conf_filters" : steps.inputs.parameters["conf_filters"],
            "fp_inputs" : steps.inputs.parameters["fp_inputs"],            
        },
        artifacts={
            "init_models": steps.inputs.artifacts["init_models"],
            "init_data": steps.inputs.artifacts["init_models"],
            "iter_data": steps.inputs.artifacts["init_models"],
        },
    )
    steps.add(block_step)

    scheduler_step = Step(
        name = name + suffix + '-scheduler',
        template=PythonOPTemplate(
            SchedulerWrapper,
            image="dflow:v1.0",
            python_packages = "..//dpgen2",
        ),
        parameters={
            "exploration_scheduler": steps.inputs.parameters['exploration_scheduler'],
            "exploration_report": block_step.outputs.parameters['exploration_report'],
        },
        artifacts={
            "trajs" : block_step.outputs.artifacts['trajs'],
        },
    )
    steps.add(scheduler_step)

    # scheduler = jsonpickle.decode(scheduler_step.outputs.parameters['exploration_scheduler'].value)
    # converged = jsonpickle.decode(scheduler_step.outputs.parameters['converged'].value)
    # stage = scheduler.get_stage()
    # iteration = scheduler.get_iteration()
    # suffix = f'-stage-{stage}-iter-{iter}'

    next_steps = Step(
        name,
        template = steps,
        parameters={
            "name_suffix" : suffix,
            "type_map" : steps.inputs.parameters["type_map"],
            "numb_models" : steps.inputs.parameters["numb_models"],
            "template_script" : steps.inputs.parameters["template_script"],
            "lmp_task_grp" : scheduler_step.outputs.parameters["lmp_task_grp"],
            "conf_selector" : scheduler_step.outputs.parameters["conf_selector"],
            "conf_filters" : steps.inputs.parameters["conf_filters"],
            "fp_inputs" : steps.inputs.parameters["fp_inputs"],
            "exploration_scheduler" : scheduler_step.outputs.parameters["exploration_scheduler"],
        },
        artifacts={
            "init_models" : block_step.outputs.artifacts['models'],
            "init_data" : steps.inputs.artifacts['init_data'],
            "iter_data" : block_step.outputs.artifacts['iter_data'],
        },
        when = "{{%s}} == false" % (scheduler_step.outputs.parameters['converged']),
    )
    steps.add(next_steps)    

    return steps


def dpgen(
        name,
        loop_op,
):    
    steps = Steps(
        name = name,
        inputs = Inputs(
            parameters={
                "name_suffix" : InputParameter(),
                "type_map" : InputParameter(),
                "numb_models": InputParameter(type=int),
                "template_script" : InputParameter(),
                "conf_filters" : InputParameter(),
                "fp_inputs" : InputParameter(),
                "exploration_scheduler" : InputParameter(),
            },
            artifacts={
                "init_models" : InputArtifact(),
                "init_data" : InputArtifact(),
                "iter_data" : InputArtifact(),
            },
        ),
    )

    scheduler_step = Step(
        name = name + '-stage-0-iter-0' + '-scheduler',
        template=PythonOPTemplate(
            SchedulerWrapper,
            image="dflow:v1.0",
            python_packages = "..//dpgen2",
        ),
        parameters={
            "exploration_scheduler": steps.inputs.parameters['exploration_scheduler'],
            "exploration_report": None,
        },
        artifacts={
            "trajs" : None,
        },
    )
    steps.add(scheduler_step)

    # scheduler = jsonpickle.decode(scheduler_step.outputs.parameters['exploration_scheduler'])
    # converged = jsonpickle.decode(scheduler_step.outputs.parameters['converged'])
    # stage = scheduler.get_stage()
    # iteration = scheduler.get_iteration()
    # suffix = f'-stage-{stage}-iter-{iter}'
    suffix = ''
    
    loop_step = Step(
        name = name,
        template = loop_op,
        parameters = {
            "name_suffix" : suffix,
            "type_map" : steps.inputs.parameters['type_map'],
            "numb_models" : steps.inputs.parameters['numb_models'],
            "template_script" : steps.inputs.parameters['template_script'],
            "lmp_task_grp" : scheduler_step.outputs.parameters['lmp_task_grp'],
            "conf_selector" : scheduler_step.outputs.parameters['conf_selector'],
            "conf_filters" : steps.inputs.parameters['conf_filters'],
            "fp_inputs" : steps.inputs.parameters['fp_inputs'],
            "exploration_scheduler" : scheduler_step.outputs.parameters['exploration_scheduler'],
        },
        artifacts={
            "init_models": steps.inputs.artifacts["init_models"],
            "init_data": steps.inputs.artifacts["init_models"],
            "iter_data": steps.inputs.artifacts["init_models"],
        },
    )
    steps.add(loop_step)
    
    return steps

    
