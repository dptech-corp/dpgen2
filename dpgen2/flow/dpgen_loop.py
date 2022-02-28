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
import pickle, jsonpickle, os
from typing import (
    List
)
from pathlib import Path
from dpgen2.exploration.scheduler import ExplorationScheduler
from dpgen2.exploration.report import ExplorationReport
from dpgen2.exploration.task import ExplorationTaskGroup
from dpgen2.exploration.selector import ConfSelector
from dpgen2.superop.block import ConcurrentLearningBlock

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
            "lmp_task_grp" : Artifact(Path),
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

        with open('lmp_task_grp.dat', 'wb') as fp:
            pickle.dump(lmp_task_grp, fp)

        return OPIO({
            "exploration_scheduler" : scheduler,
            "converged" : conv,
            "conf_selector" : selector,
            "lmp_task_grp" : Path('lmp_task_grp.dat'),
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


class ConcurrentLearningLoop(Steps):
    def __init__(
            self,
            name : str,
            block_op : Steps,
            image : str = "dflow:v1.0",
            upload_python_package : str = None,
    ):
        self._input_parameters={
            "block_id" : InputParameter(),
            "type_map" : InputParameter(),
            "numb_models": InputParameter(type=int),
            "template_script" : InputParameter(),
            "train_config" : InputParameter(),
            "lmp_config" : InputParameter(),
            "conf_selector" : InputParameter(),
            "fp_inputs" : InputParameter(),
            "fp_config" : InputParameter(),
            "exploration_scheduler" : InputParameter(),
        }
        self._input_artifacts={
            "init_models" : InputArtifact(),
            "init_data" : InputArtifact(),
            "iter_data" : InputArtifact(),
            "lmp_task_grp" : InputArtifact(),
        }
        self._output_parameters={
            "exploration_scheduler": OutputParameter(),
        }
        self._output_artifacts={
            "models": OutputArtifact(),
            "iter_data" : OutputArtifact(),
        }
        
        super().__init__(
            name = name,
            inputs = Inputs(
                parameters=self._input_parameters,
                artifacts=self._input_artifacts,
            ),
            outputs=Outputs(
                parameters=self._output_parameters,
                artifacts=self._output_artifacts,
            ),
        )

        self._my_keys = ['block', 'scheduler', 'id']
        self._keys = \
            self._my_keys[:1] + \
            block_op.keys + \
            self._my_keys[1:3]
        self.step_keys = {}
        for ii in self._my_keys:
            self.step_keys[ii] = '--'.join(
                ["%s"%self.inputs.parameters["block_id"], ii]
            )
        
        self = _loop(
            self,
            self.step_keys,
            name,
            block_op,
            image = image,
            upload_python_package = upload_python_package,
        )

    @property
    def input_parameters(self):
        return self._input_parameters

    @property
    def input_artifacts(self):
        return self._input_artifacts

    @property
    def output_parameters(self):
        return self._output_parameters

    @property
    def output_artifacts(self):
        return self._output_artifacts

    @property
    def keys(self):
        return self._keys


class ConcurrentLearning(Steps):
    def __init__(
            self,
            name : str,
            block_op : Steps,
            image : str = "dflow:v1.0",
            upload_python_package : str = None,
    ):
        self.loop = ConcurrentLearningLoop(
            name+'-loop',
            block_op,
            image = image,
            upload_python_package = upload_python_package,
        )
        
        self._input_parameters={
            "type_map" : InputParameter(),
            "numb_models": InputParameter(type=int),
            "template_script" : InputParameter(),
            "train_config" : InputParameter(),
            "lmp_config" : InputParameter(),
            "fp_inputs" : InputParameter(),
            "fp_config" : InputParameter(),
            "exploration_scheduler" : InputParameter(),
        }
        self._input_artifacts={
            "init_models" : InputArtifact(),
            "init_data" : InputArtifact(),
            "iter_data" : InputArtifact(),
        }
        self._output_parameters={
            "exploration_scheduler": OutputParameter(),
        }
        self._output_artifacts={
            "models": OutputArtifact(),
            "iter_data" : OutputArtifact(),
        }        
        
        super().__init__(
            name = name,
            inputs = Inputs(
                parameters=self._input_parameters,
                artifacts=self._input_artifacts,
            ),
            outputs=Outputs(
                parameters=self._output_parameters,
                artifacts=self._output_artifacts,
            ),
        )

        self._init_keys = ['scheduler', 'id']
        self.loop_key = 'loop'
        self.step_keys = {}
        for ii in self._init_keys:
            self.step_keys[ii] = '--'.join(['init', ii])

        self = _dpgen(
            self,
            self.step_keys,
            name, 
            self.loop,
            self.loop_key,
            image = image,
            upload_python_package = upload_python_package,
        )

    @property
    def input_parameters(self):
        return self._input_parameters

    @property
    def input_artifacts(self):
        return self._input_artifacts

    @property
    def output_parameters(self):
        return self._output_parameters

    @property
    def output_artifacts(self):
        return self._output_artifacts

    @property
    def init_keys(self):
        return self._init_keys

    @property
    def loop_keys(self):
        return [self.loop_key] + self.loop.keys


def _loop (
        steps, 
        step_keys,
        name : str,
        block_op : OP,
        image : str = "dflow:v1.0",
        upload_python_package : str = None,
):    
    block_step = Step(
        name = name + '-block',
        template = block_op,
        parameters={
            "block_id" : steps.inputs.parameters["block_id"],
            "type_map" : steps.inputs.parameters["type_map"],
            "numb_models" : steps.inputs.parameters["numb_models"],
            "template_script" : steps.inputs.parameters["template_script"],
            "train_config" : steps.inputs.parameters["train_config"],
            "lmp_config" : steps.inputs.parameters["lmp_config"],
            "conf_selector" : steps.inputs.parameters["conf_selector"],
            "fp_inputs" : steps.inputs.parameters["fp_inputs"],            
            "fp_config" : steps.inputs.parameters["fp_config"],
        },
        artifacts={
            "lmp_task_grp" : steps.inputs.artifacts["lmp_task_grp"],
            "init_models": steps.inputs.artifacts["init_models"],
            "init_data": steps.inputs.artifacts["init_data"],
            "iter_data": steps.inputs.artifacts["iter_data"],
        },
        key = step_keys['block'],
    )
    steps.add(block_step)

    scheduler_step = Step(
        name = name + '-scheduler',
        template=PythonOPTemplate(
            SchedulerWrapper,
            image=image,
            python_packages = upload_python_package,
        ),
        parameters={
            "exploration_scheduler": steps.inputs.parameters['exploration_scheduler'],
            "exploration_report": block_step.outputs.parameters['exploration_report'],
        },
        artifacts={
            "trajs" : block_step.outputs.artifacts['trajs'],
        },
        key = step_keys['scheduler'],
    )
    steps.add(scheduler_step)

    id_step = Step(
        name = name + '-make-block-id',
        template=PythonOPTemplate(
            MakeBlockId,
            image=image,
            python_packages = upload_python_package,
        ),
        parameters={
            "exploration_scheduler": scheduler_step.outputs.parameters['exploration_scheduler'],
        },
        artifacts={
        },
        key = step_keys['id'],
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
            "lmp_config" : steps.inputs.parameters["lmp_config"],
            "conf_selector" : scheduler_step.outputs.parameters["conf_selector"],
            "fp_inputs" : steps.inputs.parameters["fp_inputs"],
            "fp_config" : steps.inputs.parameters["fp_config"],
            "exploration_scheduler" : scheduler_step.outputs.parameters["exploration_scheduler"],
        },
        artifacts={
            "lmp_task_grp" : scheduler_step.outputs.artifacts["lmp_task_grp"],
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


def _dpgen(
        steps, 
        step_keys,
        name,
        loop_op,
        loop_key,
        image = "dflow:v1.0",
        upload_python_package : str = None
):    
    scheduler_step = Step(
        name = name + '-scheduler',
        template=PythonOPTemplate(
            SchedulerWrapper,
            image=image,
            python_packages = upload_python_package,
        ),
        parameters={
            "exploration_scheduler": steps.inputs.parameters['exploration_scheduler'],
            "exploration_report": None,
        },
        artifacts={
            "trajs" : None,
        },
        key = step_keys['scheduler'],
    )
    steps.add(scheduler_step)

    id_step = Step(
        name = name + '-make-block-id',
        template=PythonOPTemplate(
            MakeBlockId,
            image=image,
            python_packages = upload_python_package,
        ),
        parameters={
            "exploration_scheduler": scheduler_step.outputs.parameters['exploration_scheduler'],
        },
        artifacts={
        },
        key = step_keys['id'],
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
            "conf_selector" : scheduler_step.outputs.parameters['conf_selector'],
            "lmp_config" : steps.inputs.parameters['lmp_config'],
            "fp_inputs" : steps.inputs.parameters['fp_inputs'],
            "fp_config" : steps.inputs.parameters['fp_config'],
            "exploration_scheduler" : scheduler_step.outputs.parameters['exploration_scheduler'],
        },
        artifacts={
            "lmp_task_grp" : scheduler_step.outputs.artifacts['lmp_task_grp'],
            "init_models": steps.inputs.artifacts["init_models"],
            "init_data": steps.inputs.artifacts["init_data"],
            "iter_data": steps.inputs.artifacts["iter_data"],
        },
        key = '--'.join(["%s"%id_step.outputs.parameters['block_id'], loop_key]),
    )
    steps.add(loop_step)

    steps.outputs.parameters["exploration_scheduler"].value_from_parameter = \
        loop_step.outputs.parameters["exploration_scheduler"]
    steps.outputs.artifacts["models"]._from = \
        loop_step.outputs.artifacts["models"]
    steps.outputs.artifacts["iter_data"]._from = \
        loop_step.outputs.artifacts["iter_data"]
    
    return steps

    
