from dflow.python import (
    OP,
    OPIO,
    OPIOSign,
    Artifact,
    upload_packages,
)

upload_packages.append(__file__)

import os, json, shutil
from pathlib import Path
from typing import Tuple, List
try:
    from context import dpgen2
except ModuleNotFoundError:
    # case of upload everything to argo, no context needed
    pass
from dpgen2.op.run_dp_train import RunDPTrain
from dpgen2.op.prep_dp_train import PrepDPTrain
from dpgen2.op.prep_lmp import PrepLmpTaskGroup
from dpgen2.op.run_lmp import RunLmp
from dpgen2.op.prep_vasp import PrepVasp
from dpgen2.op.run_vasp import RunVasp
from dpgen2.op.collect_data import CollectData
from dpgen2.op.select_confs import SelectConfs
from dpgen2.utils.conf_selector import ConfSelector
from dpgen2.utils.conf_filter import ConfFilter
from dpgen2.exploration.report import ExplorationReport

class MockedPrepDPTrain(PrepDPTrain):
    @OP.exec_sign_check
    def execute(
            self,
            ip : OPIO,
    ) -> OPIO:
        template = ip['template_script']
        numb_models = ip['numb_models']
        ofiles = []
        osubdirs = []

        for ii in range(numb_models):
            jtmp = template
            jtmp['seed'] = ii
            subdir = Path(f'task.{ii:04d}') 
            subdir.mkdir(exist_ok=True, parents=True)
            fname = subdir / 'input.json'
            with open(fname, 'w') as fp:
                json.dump(jtmp, fp, indent = 4)
            osubdirs.append(str(subdir))
            ofiles.append(fname)

        op = OPIO({
            "task_names" : osubdirs,
            "task_paths" : [Path(ii) for ii in osubdirs],
        })
        return op


class MockedRunDPTrain(RunDPTrain):
    @OP.exec_sign_check
    def execute(
            self,
            ip : OPIO,
    ) -> OPIO:
        work_dir = Path(ip['task_name'])
        script = ip['task_path'] / 'input.json'
        init_model = Path(ip['init_model'])
        init_data = ip['init_data']
        iter_data = ip['iter_data']

        script = Path(script).resolve()
        init_model = init_model.resolve()
        init_model_str = str(init_model)
        init_data = [ii.resolve() for ii in init_data]
        iter_data = [ii.resolve() for ii in iter_data]
        init_data_str = [str(ii) for ii in init_data]
        iter_data_str = [str(ii) for ii in iter_data]

        with open(script) as fp:
            jtmp = json.load(fp)        
        data = []
        for ii in sorted(init_data_str):
            data.append(ii)
        for ii in sorted(iter_data_str):
            data.append(ii)
        jtmp['data'] = data
        with open(script, 'w') as fp:
            json.dump(jtmp, fp, indent=4)

        cwd = os.getcwd()
        work_dir.mkdir(exist_ok=True, parents=True)
        os.chdir(work_dir)

        oscript = Path('input.json')
        if not oscript.exists():
            from shutil import copyfile
            copyfile(script, oscript)
        model = Path('model.pb')
        lcurve = Path('lcurve.out')
        log = Path('log')

        assert(init_model.exists())        
        with log.open("w") as f:
            f.write(f'init_model {str(init_model)} OK\n')
        for ii in jtmp['data']:
            assert(Path(ii).exists())
            assert((ii in init_data_str) or (ii in iter_data_str))
            with log.open("a") as f:
                f.write(f'data {str(ii)} OK\n')
        assert(script.exists())
        with log.open("a") as f:
            f.write(f'script {str(script)} OK\n')

        with model.open("w") as f:
            f.write('read from init model: \n')
            f.write(init_model.read_text() + '\n')
        with lcurve.open("w") as f:
            f.write('read from train_script: \n')
            f.write(script.read_text() + '\n')

        os.chdir(cwd)
        
        return OPIO({
            'script' : work_dir/oscript,
            'model' : work_dir/model,
            'lcurve' : work_dir/lcurve,
            'log' : work_dir/log
        })


class MockedRunLmp(RunLmp):
    @OP.exec_sign_check
    def execute(
            self,
            ip : OPIO,
    ) -> OPIO:
        task_name = ip['task_name']
        task_path = ip['task_path']
        models = ip['models']
        
        task_path = task_path.resolve()
        models = [ii.resolve() for ii in models]
        models_str = [str(ii) for ii in models]
        
        work_dir = Path(task_name)

        cwd = os.getcwd()
        work_dir.mkdir(exist_ok=True, parents=True)
        os.chdir(work_dir)

        import glob
        ifiles = glob.glob(str(task_path / '*'))
        for ii in ifiles:
            if not Path(Path(ii).name).exists():
                Path(Path(ii).name).symlink_to(ii)
        for ii in models:
            if not Path(Path(ii).name).exists():
                Path(Path(ii).name).symlink_to(ii)
        
        log = Path('log')
        traj = Path('dump.traj')
        model_devi = Path('model_devi.out')
        
        # fc = ['log of {task_name}']
        # for ii in ['conf.lmp', 'in.lammps'] + models_str:
        #     if Path(ii).exists():
        #         fc.append(f'{ii} OK')
        # log.write_text('\n'.join(fc))        
        # log.write_text('log of {task_name}')
        fc = []
        for ii in ['conf.lmp', 'in.lammps'] + [ii.name for ii in models]:
             fc.append(Path(ii).read_text())
        log.write_text('\n'.join(fc))
        traj.write_text(f'traj of {task_name}')
        model_devi.write_text(f'model_devi of {task_name}')

        os.chdir(cwd)

        return OPIO({
            'log' : work_dir/log,
            'traj' : work_dir/traj,
            'model_devi' : work_dir/model_devi,
        })


class MockedPrepVasp(PrepVasp):
    @OP.exec_sign_check
    def execute(
            self,
            ip : OPIO,
    ) -> OPIO:
        confs = ip['confs']
        # incar_temp = ip['incar_temp']
        # potcars = ip['potcars']
        vasp_input = ip['inputs']
        incar_temp = vasp_input.incar_temp
        potcars = vasp_input.potcars

        nconfs = len(confs)
        task_paths = []

        for ii in range(nconfs):
            task_path = Path(f'task.{ii:06d}')
            task_path.mkdir(exist_ok=True, parents=True)
            from shutil import copyfile
            copyfile(confs[ii], task_path/'POSCAR')
            (task_path/'INCAR').write_text(incar_temp)
            task_paths.append(task_path)

        task_names = [str(ii) for ii in task_paths]
        return OPIO({
            'task_names' : task_names,
            'task_paths' : task_paths,
        })


class MockedRunVasp(RunVasp):
    @OP.exec_sign_check
    def execute(
            self,
            ip : OPIO,
    ) -> OPIO:
        task_name = ip['task_name']
        task_path = ip['task_path']

        work_dir = Path(task_name)

        cwd = os.getcwd()
        work_dir.mkdir(exist_ok=True, parents=True)
        os.chdir(work_dir)

        import glob
        ifiles = glob.glob(str(task_path / '*'))
        for ii in ifiles:
            if not Path(Path(ii).name).exists():
                Path(Path(ii).name).symlink_to(ii)
        
        log = Path('log')
        # labeled_data = Path('labeled_data')
        labeled_data = Path('data_'+task_name)
        
        fc = []
        for ii in ['POSCAR', 'INCAR']:
             fc.append(Path(ii).read_text())
        log.write_text('\n'.join(fc))
        labeled_data.mkdir(exist_ok=True, parents=True)
        (labeled_data / 'data').write_text(f'labeled_data of {task_name}')

        os.chdir(cwd)

        return OPIO({
            'log' : work_dir/log,
            'labeled_data' : work_dir/labeled_data,
        })


class MockedCollectData(CollectData):
    @OP.exec_sign_check
    def execute(
            self,
            ip : OPIO,
    ) -> OPIO:
        name = ip['name']
        labeled_data = ip['labeled_data']
        iter_data = ip['iter_data']

        new_iter_data = set()
        # copy iter_data
        for ii in iter_data:
            iiname = ii.name
            shutil.copytree(ii, iiname)
            new_iter_data.add(Path(iiname))

        # collect labled data
        name = Path(name)
        name.mkdir(exist_ok=True, parents=True)
        
        for ii in labeled_data:
            iiname = ii.name
            shutil.copytree(ii, name/iiname)
        new_iter_data.add(name)
        
        return OPIO({
            "iter_data" : new_iter_data,
        })


class MockedExplorationReport(ExplorationReport):
    def __init__(self):
        pass

    def converged (
            self, 
    ) -> bool :
        return True

    def failed_ratio (
            self, 
            tag = None,
    ) -> float :
        return 0.

    def accurate_ratio (
            self,
            tag = None,
    ) -> float :
        return 1.

    def candidate_ratio (
            self,
            tag = None,
    ) -> float :
        return 0.

    def update_trust_levels (
            self,
    ) -> Tuple[float] :
        return (0.1, 0.2, 0.1, 0.2)


class MockedConfSelector(ConfSelector):
    def select (
            self,
            trajs : List[Path],
            model_devis : List[Path],
            conf_filters : List[ConfFilter] = [],
            traj_fmt : str = 'deepmd/npy',
            type_map : List[str] = None,
    ) -> List[ Path ] :
        confs = []
        fname = Path('conf.0')
        fname.write_text('conf of conf.0')
        confs.append(fname)
        fname = Path('conf.1')
        fname.write_text('conf of conf.1')
        confs.append(fname)
        return confs

class MockedSelectConfs(SelectConfs):
    @OP.exec_sign_check
    def execute(
            self,
            ip : OPIO,
    ) -> OPIO:
        conf_selector = ip['conf_selector']
        trajs = ip['trajs']
        model_devis = ip['model_devis']
        confs = conf_selector.select(trajs, model_devis)
        report = MockedExplorationReport()

        return OPIO({
            "report" : report,
            "confs" : confs,
        })
