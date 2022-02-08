from . import ExplorationReport

class NaiveExplorationReport(ExplorationReport):
    def __init__(
            self,
            counter_f,
            counter_v,
    ):
        self.valid_quantities = ['force', 'virial']
        self.valid_items = ['accurate', 'candidate', 'failed']
        self.report = {
            'force' : {},
            'virial' : {},
        }
        self.report['force']['candidate'], \
        self.report['force']['accurate'], \
        self.report['force']['failed'] = \
            NaiveExplorationReport.calculate_ratio(
                counter_f['candidate'], 
                counter_f['accurate'], 
                counter_f['failed']
            )
        self.report['virial']['candidate'], \
        self.report['virial']['accurate'], \
        self.report['virial']['failed'] = \
            NaiveExplorationReport.calculate_ratio(
                counter_v['candidate'], 
                counter_v['accurate'], 
                counter_v['failed']
            )

    @staticmethod
    def calculate_ratio(
            cc,
            ca,
            cf,
    ):
        sumed = float(cc + ca + cf)
        if sumed > 0:
            rf = float(cf) / sumed
            rc = float(cc) / sumed
            ra = float(ca) / sumed
        else:
            rf = rc = ra = None
        return rc, ra, rf

    def failed_ratio (
            self, 
            tag = None,
    ) -> float :
        return self.ratio('force', 'failed')

    def accurate_ratio (
            self,
            tag = None,
    ) -> float :
        return self.ratio('force', 'accurate')

    def candidate_ratio (
            self,
            tag = None,
    ) -> float :
        return self.ratio('force', 'candidate')

    def ratio(
            self,
            quantity : str,
            item : str
    )-> float:
        if not quantity in self.valid_quantities:
            raise RuntimeError(f'invalid quantity {quantity}, must in [{" ".join(self.valid_quantities)}]')
        if not item in self.valid_items:
            raise RuntimeError(f'invalid item {item}, must in [{" ".join(self.valid_items)}]')
        return self.report[quantity][item]


