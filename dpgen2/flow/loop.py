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
    if_expression,
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


class MakeBlockId(OP):
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "exploration_scheduler" : ExplorationScheduler,
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "block_id" : str,
        })

    @OP.exec_sign_check
    def execute(
            self,
            ip : OPIO,
    ) -> OPIO:
        scheduler = ip['exploration_scheduler']
        
        stage = scheduler.get_stage()
        iteration = scheduler.get_iteration()

        return OPIO({
            "block_id" : f'iter-{iteration:06d}',
        })



def loop (
        name : str,
        block_op : OP,
):
    steps = Steps(
        name = name,
        inputs = Inputs(
            parameters={
                "block_id" : InputParameter(),
                "type_map" : InputParameter(),
                "numb_models": InputParameter(type=int),
                "template_script" : InputParameter(),
                "train_config" : InputParameter(),
                "lmp_task_grp" : InputParameter(),
                "conf_selector" : InputParameter(),
                "fp_inputs" : InputParameter(),
                "exploration_scheduler" : InputParameter(),
            },
            artifacts={
                "init_models" : InputArtifact(),
                "init_data" : InputArtifact(),
                "iter_data" : InputArtifact(),
            },
        ),
        outputs=Outputs(
            parameters={
                "exploration_scheduler": OutputParameter(),
            },
            artifacts={
                "models": OutputArtifact(),
                "iter_data" : OutputArtifact(),
            },
        ),
    )
    
    # suffix = steps.inputs.parameters["name_suffix"].value
    suffix=''

    block_step = Step(
        name = name + suffix + '-block',
        template = block_op,
        parameters={
            "block_id" : steps.inputs.parameters["block_id"],
            "type_map" : steps.inputs.parameters["type_map"],
            "numb_models" : steps.inputs.parameters["numb_models"],
            "template_script" : steps.inputs.parameters["template_script"],
            "train_config" : steps.inputs.parameters["train_config"],
            "lmp_task_grp" : steps.inputs.parameters["lmp_task_grp"],
            "conf_selector" : steps.inputs.parameters["conf_selector"],
            "fp_inputs" : steps.inputs.parameters["fp_inputs"],            
        },
        artifacts={
            "init_models": steps.inputs.artifacts["init_models"],
            "init_data": steps.inputs.artifacts["init_data"],
            "iter_data": steps.inputs.artifacts["iter_data"],
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

    id_step = Step(
        name = name + '-make-block-id',
        template=PythonOPTemplate(
            MakeBlockId,
            image="dflow:v1.0",
            python_packages = "..//dpgen2",
        ),
        parameters={
            "exploration_scheduler": scheduler_step.outputs.parameters['exploration_scheduler'],
        },
        artifacts={
        },
    )
    steps.add(id_step)

    next_step = Step(
        name = name+'-next',
        template = steps,
        parameters={
            "block_id" : id_step.outputs.parameters['block_id'],
            "type_map" : steps.inputs.parameters["type_map"],
            "numb_models" : steps.inputs.parameters["numb_models"],
            "template_script" : steps.inputs.parameters["template_script"],
            "train_config" : steps.inputs.parameters["train_config"],
            "lmp_task_grp" : scheduler_step.outputs.parameters["lmp_task_grp"],
            "conf_selector" : scheduler_step.outputs.parameters["conf_selector"],
            "fp_inputs" : steps.inputs.parameters["fp_inputs"],
            "exploration_scheduler" : scheduler_step.outputs.parameters["exploration_scheduler"],
        },
        artifacts={
            "init_models" : block_step.outputs.artifacts['models'],
            "init_data" : steps.inputs.artifacts['init_data'],
            "iter_data" : block_step.outputs.artifacts['iter_data'],
        },
        when = "%s == false" % (scheduler_step.outputs.parameters['converged']),
    )
    steps.add(next_step)    

    steps.outputs.parameters['exploration_scheduler'].value_from_expression = \
        if_expression(
            _if = (scheduler_step.outputs.parameters['converged'] == True),
            _then = scheduler_step.outputs.parameters['exploration_scheduler'],
            _else = next_step.outputs.parameters['exploration_scheduler'],
        )
    steps.outputs.artifacts['models'].from_expression = \
        if_expression(
            _if = (scheduler_step.outputs.parameters['converged'] == True),
            _then = block_step.outputs.artifacts['models'],
            _else = next_step.outputs.artifacts['models'],
        )
    steps.outputs.artifacts['iter_data'].from_expression = \
        if_expression(
            _if = (scheduler_step.outputs.parameters['converged'] == True),
            _then = block_step.outputs.artifacts['iter_data'],
            _else = next_step.outputs.artifacts['iter_data'],
        )

    return steps


def dpgen(
        name,
        loop_op,
):    
    steps = Steps(
        name = name,
        inputs = Inputs(
            parameters={
                "type_map" : InputParameter(),
                "numb_models": InputParameter(type=int),
                "template_script" : InputParameter(),
                "train_config" : InputParameter(),
                "fp_inputs" : InputParameter(),
                "exploration_scheduler" : InputParameter(),
            },
            artifacts={
                "init_models" : InputArtifact(),
                "init_data" : InputArtifact(),
                "iter_data" : InputArtifact(),
            },
        ),
        outputs=Outputs(
            parameters={
                "exploration_scheduler": OutputParameter(),
            },
            artifacts={
                "models": OutputArtifact(),
                "iter_data" : OutputArtifact(),
            },
        ),
    )

    scheduler_step = Step(
        name = name + '-scheduler',
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

    id_step = Step(
        name = name + '-make-block-id',
        template=PythonOPTemplate(
            MakeBlockId,
            image="dflow:v1.0",
            python_packages = "..//dpgen2",
        ),
        parameters={
            "exploration_scheduler": scheduler_step.outputs.parameters['exploration_scheduler'],
        },
        artifacts={
        },
    )
    steps.add(id_step)

    loop_step = Step(
        name = name + '-loop',
        template = loop_op,
        parameters = {
            "block_id" : id_step.outputs.parameters['block_id'],
            "type_map" : steps.inputs.parameters['type_map'],
            "numb_models" : steps.inputs.parameters['numb_models'],
            "template_script" : steps.inputs.parameters['template_script'],
            "train_config" : steps.inputs.parameters['train_config'],
            "lmp_task_grp" : scheduler_step.outputs.parameters['lmp_task_grp'],
            "conf_selector" : scheduler_step.outputs.parameters['conf_selector'],
            "fp_inputs" : steps.inputs.parameters['fp_inputs'],
            "exploration_scheduler" : scheduler_step.outputs.parameters['exploration_scheduler'],
        },
        artifacts={
            "init_models": steps.inputs.artifacts["init_models"],
            "init_data": steps.inputs.artifacts["init_data"],
            "iter_data": steps.inputs.artifacts["iter_data"],
        },
    )
    steps.add(loop_step)

    steps.outputs.parameters["exploration_scheduler"].value_from_parameter = \
        loop_step.outputs.parameters["exploration_scheduler"]
    steps.outputs.artifacts["models"]._from = \
        loop_step.outputs.artifacts["models"]
    steps.outputs.artifacts["iter_data"]._from = \
        loop_step.outputs.artifacts["iter_data"]
    
    return steps

    
